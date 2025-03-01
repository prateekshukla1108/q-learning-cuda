#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
}

struct EnvironmentConfig {
    int rows;
    int cols;
    int startState;
    int goalState;
    float rewardEmpty;
    float rewardObstacle;
    float rewardGoal;
    float rewardSlow;
    float rewardBoundary;
    float rewardStep;
    float obstacleProb;
    float slowProb;
};

class GridWorld {
public:
    int rows, cols, numStates;
    std::vector<int> grid;
    EnvironmentConfig config;
    int startState, goalState;

    GridWorld(const EnvironmentConfig& envConfig) : config(envConfig) {
        if(config.obstacleProb + config.slowProb > 1.0f) {
            std::cerr << "Error: obstacleProb + slowProb exceeds 1.0" << std::endl;
            exit(1);
        }
        rows = config.rows;
        cols = config.cols;
        numStates = rows * cols;
        grid.resize(numStates, 0);
        startState = config.startState;
        goalState = config.goalState;
        grid[startState] = 3;
        grid[goalState] = 2;
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < numStates; i++) {
            if(i == startState || i == goalState) continue;
            float prob = dist(rng);
            if(prob < config.obstacleProb) grid[i] = 1;
            else if(prob < config.obstacleProb + config.slowProb) grid[i] = 4;
            else grid[i] = 0;
        }
    }

    int computeIntendedNextState(int currentState, int action) const {
        int row = currentState / cols;
        int col = currentState % cols;
        int new_row = row, new_col = col;
        if(action == 0) new_row = row - 1;
        else if(action == 1) new_row = row + 1;
        else if(action == 2) new_col = col - 1;
        else if(action == 3) new_col = col + 1;
        else if(action == 4) { new_row = row - 1; new_col = col - 1; }
        else if(action == 5) { new_row = row - 1; new_col = col + 1; }
        else if(action == 6) { new_row = row + 1; new_col = col - 1; }
        else if(action == 7) { new_row = row + 1; new_col = col + 1; }
        if(new_row < 0 || new_row >= rows || new_col < 0 || new_col >= cols) return -1;
        return new_row * cols + new_col;
    }

    int step(int currentState, int action) const {
        int intended = computeIntendedNextState(currentState, action);
        if (intended == -1) return currentState;
        if (grid[intended] == 1) return currentState;
        return intended;
    }

    float getReward(int currentState, int action, int nextState) const {
        int intended = computeIntendedNextState(currentState, action);
        if (currentState == nextState) {
            if (intended == -1) return config.rewardBoundary;
            else if (grid[intended] == 1) return config.rewardObstacle;
            else return config.rewardStep;
        } else {
            int cellType = grid[nextState];
            if(cellType == 0) return config.rewardEmpty;
            else if(cellType == 2) return config.rewardGoal;
            else if(cellType == 4) return config.rewardSlow;
            else return config.rewardStep;
        }
    }
};

struct Transition {
    int state;
    int action;
    float reward;
    int nextState;
    bool done;
};

__global__ void updateQKernelBatch(const int *states, const int *actions, const float *rewards,
                                     const int *nextStates, const int *dones, float *d_onlineQ,
                                     const float *d_targetQ, float alpha, float gamma, int numActions, int batchSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < batchSize) {
        int s = states[i];
        int a = actions[i];
        float r = rewards[i];
        int s_next = nextStates[i];
        int done = dones[i];
        float maxQ = -1e20f;
        for (int j = 0; j < numActions; j++) {
            float q = d_targetQ[s_next * numActions + j];
            if(q > maxQ) { maxQ = q; }
        }
        float target = r;
        if(!done) {
            target += gamma * maxQ;
        }
        int index = s * numActions + a;
        int *addr_as_i = (int*)&d_onlineQ[index];
        int oldInt = *addr_as_i;
        float oldVal = __int_as_float(oldInt);
        int newInt;
        float newVal;
        while (true) {
            newVal = oldVal + alpha * (target - oldVal);
            newInt = __float_as_int(newVal);
            int assumed = atomicCAS(addr_as_i, oldInt, newInt);
            if (assumed == oldInt) break;
            oldInt = assumed;
            oldVal = __int_as_float(oldInt);
        }
    }
}

class QLearningAgent {
public:
    int numStates, numActions;
    float alpha, gamma, epsilon, minEpsilon, epsilonDecay;
    int updateTargetEvery, replayBatchSize, replayCapacity;
    std::vector<Transition> replayBuffer;
    float *d_onlineQ, *d_targetQ;

    QLearningAgent(int numStates, int numActions, float alpha, float gamma, float initialEpsilon,
                   float minEpsilon, float epsilonDecay, int updateTargetEvery,
                   int replayCapacity, int replayBatchSize)
    : numStates(numStates), numActions(numActions), alpha(alpha), gamma(gamma),
      epsilon(initialEpsilon), minEpsilon(minEpsilon), epsilonDecay(epsilonDecay),
      updateTargetEvery(updateTargetEvery), replayCapacity(replayCapacity), replayBatchSize(replayBatchSize)
    {
        CUDA_CHECK(cudaMalloc(&d_onlineQ, numStates * numActions * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targetQ, numStates * numActions * sizeof(float)));
        std::vector<float> qInit(numStates * numActions);
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        for(auto &val : qInit) { val = dist(rng); }
        CUDA_CHECK(cudaMemcpy(d_onlineQ, qInit.data(), numStates * numActions * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targetQ, qInit.data(), numStates * numActions * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~QLearningAgent() {
        CUDA_CHECK(cudaFree(d_onlineQ));
        CUDA_CHECK(cudaFree(d_targetQ));
    }

    int chooseAction(int state) {
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float randomValue = dist(rng);
        if(randomValue < epsilon) {
            std::uniform_int_distribution<int> actionDist(0, numActions - 1);
            return actionDist(rng);
        } else {
            std::vector<float> qValues(numActions);
            CUDA_CHECK(cudaMemcpy(qValues.data(), d_onlineQ + state * numActions, numActions * sizeof(float), cudaMemcpyDeviceToHost));
            int bestAction = 0;
            float bestValue = qValues[0];
            for (int a = 1; a < numActions; a++) {
                if(qValues[a] > bestValue) { bestValue = qValues[a]; bestAction = a; }
            }
            return bestAction;
        }
    }

    void storeTransition(const Transition &trans) {
        if(replayBuffer.size() >= (size_t)replayCapacity) {
            replayBuffer.erase(replayBuffer.begin());
        }
        replayBuffer.push_back(trans);
    }

    void trainOnBatch() {
        if(replayBuffer.size() < (size_t)replayBatchSize) return;
        std::vector<int> states(replayBatchSize);
        std::vector<int> actions(replayBatchSize);
        std::vector<float> rewards(replayBatchSize);
        std::vector<int> nextStates(replayBatchSize);
        std::vector<int> dones(replayBatchSize);
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<size_t> dist(0, replayBuffer.size() - 1);
        for(int i = 0; i < replayBatchSize; i++){
            size_t index = dist(rng);
            const Transition &trans = replayBuffer[index];
            states[i] = trans.state;
            actions[i] = trans.action;
            rewards[i] = trans.reward;
            nextStates[i] = trans.nextState;
            dones[i] = trans.done ? 1 : 0;
        }
        int *d_states, *d_actions, *d_nextStates, *d_dones;
        float *d_rewards;
        CUDA_CHECK(cudaMalloc(&d_states, replayBatchSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_actions, replayBatchSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rewards, replayBatchSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nextStates, replayBatchSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dones, replayBatchSize * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_states, states.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_actions, actions.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rewards, rewards.data(), replayBatchSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nextStates, nextStates.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dones, dones.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice));
        int threads = 256;
        int blocks = (replayBatchSize + threads - 1) / threads;
        updateQKernelBatch<<<blocks, threads>>>(d_states, d_actions, d_rewards, d_nextStates, d_dones,
                                                  d_onlineQ, d_targetQ, alpha, gamma, numActions, replayBatchSize);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_actions));
        CUDA_CHECK(cudaFree(d_rewards));
        CUDA_CHECK(cudaFree(d_nextStates));
        CUDA_CHECK(cudaFree(d_dones));
    }

    void updateTargetNetwork() {
        CUDA_CHECK(cudaMemcpy(d_targetQ, d_onlineQ, numStates * numActions * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    void decayEpsilon() {
        epsilon = std::max(minEpsilon, epsilon * epsilonDecay);
    }
};

int main(){
    EnvironmentConfig envConfig;
    envConfig.rows = 10;
    envConfig.cols = 10;
    envConfig.startState = 0;
    envConfig.goalState = envConfig.rows * envConfig.cols - 1;
    envConfig.rewardEmpty = -1.0f;
    envConfig.rewardObstacle = -50.0f;
    envConfig.rewardGoal = 100.0f;
    envConfig.rewardSlow = -3.0f;
    envConfig.rewardBoundary = -10.0f;
    envConfig.rewardStep = -1.0f;
    envConfig.obstacleProb = 0.2f;
    envConfig.slowProb = 0.1f;

    GridWorld gridWorld(envConfig);
    int numStates = gridWorld.numStates;
    int numActions = 8;
    int numEpisodes = 2000;
    int maxStepsPerEpisode = 200;

    QLearningAgent agent(numStates, numActions, 0.1f, 0.9f, 0.9f, 0.1f, 0.995f, 50, 10000, 32);

    std::ofstream logFile("training_log.csv");
    logFile << "Episode,Steps,TotalReward,FinalState,Epsilon\n";

    for (int episode = 0; episode < numEpisodes; episode++){
        int currentState = gridWorld.startState;
        float totalReward = 0.0f;
        bool done = false;
        int stepsTaken = 0;
        for (; stepsTaken < maxStepsPerEpisode && !done; stepsTaken++){
            int action = agent.chooseAction(currentState);
            int nextState = gridWorld.step(currentState, action);
            float reward = gridWorld.getReward(currentState, action, nextState);
            if (nextState == gridWorld.goalState) done = true;
            Transition trans { currentState, action, reward, nextState, done };
            agent.storeTransition(trans);
            totalReward += reward;
            currentState = nextState;
        }
        agent.trainOnBatch();
        if (episode % agent.updateTargetEvery == 0) {
            agent.updateTargetNetwork();
        }
        agent.decayEpsilon();
        logFile << episode << "," << (stepsTaken) << "," << totalReward << "," << currentState << "," << agent.epsilon << "\n";
        std::cout << "Episode " << episode << ": Steps = " << stepsTaken << ", Total Reward = " << totalReward
                  << ", Final State = " << currentState << ", Epsilon = " << agent.epsilon << "\n";
    }
    logFile.close();

    std::ofstream plotScript("plot_results.py");
    plotScript << "import pandas as pd\n";
    plotScript << "import matplotlib.pyplot as plt\n";
    plotScript << "data = pd.read_csv('training_log.csv')\n";
    plotScript << "plt.figure(); plt.plot(data['Episode'], data['TotalReward']); plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.title('Episode Reward Curve'); plt.savefig('episode_reward.png')\n";
    plotScript << "plt.figure(); plt.plot(data['Episode'], data['Steps']); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.title('Steps per Episode'); plt.savefig('steps_per_episode.png')\n";
    plotScript << "plt.figure(); plt.plot(data['Episode'], data['Epsilon']); plt.xlabel('Episode'); plt.ylabel('Epsilon'); plt.title('Epsilon Decay Curve'); plt.savefig('epsilon_decay.png')\n";
    plotScript.close();

    system("python plot_results.py");
    return 0;
}

