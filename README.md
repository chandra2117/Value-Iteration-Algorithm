# Value-Iteration-Algorithm
## AIM

To implement and demonstrate the Value Iteration algorithm for finding the optimal value function and optimal policy for a given Markov Decision Process (MDP). Use the implementation to compute the optimal policy, report the optimal value function, and evaluate the success rate of the obtained policy on the environment.


## PROBLEM STATEMENT

Given a finite Markov Decision Process defined by:

a set of states S (|S| = number of states),

a set of actions A (|A| = number of actions),

transition dynamics P[s][a] = [(prob, next_state, reward, done), ...] for each state s and action a,

a discount factor gamma (0 < gamma <= 1),

we want to compute:

The optimal value function V* that gives the maximum expected discounted return from each state when acting optimally.

The optimal policy pi* that chooses for every state the action maximizing expected return.

The empirical success rate of running pi* on the environment (fraction of episodes that reach the terminal success condition).

This task is solved using Value Iteration: an iterative dynamic programming algorithm that repeatedly updates state values with the Bellman optimality operator until convergence.

## VALUE ITERATION ALGORITHM (Steps)

Initialize the value function V(s) arbitrarily (commonly zeros) for all states s.

Repeat until the change in V is below a small threshold theta:

For each state s, compute Q(s,a) = sum_{s',r} P(s'|s,a) * [r + gamma * V(s')] for every action a.

Update V(s) := max_a Q(s,a).

Track the maximum change delta = max_s |V_old(s) - V_new(s)| and stop when delta < theta.

Extract policy: for each state s, set pi(s) = argmax_a sum_{s',r} P(s'|s,a) * [r + gamma * V(s')].

(Optional) Evaluate policy: run episodes using pi in the environment and compute the success rate (successful terminal episodes / total episodes).

Convergence guarantee: For a finite MDP and 0 <= gamma < 1, Value Iteration converges to the unique optimal value function V*. For gamma = 1 it converges if the MDP is episodic (guaranteed termination) and appropriate conditions hold.

## VALUE ITERATION FUNCTION
```

envdesc  = ['SHFH','FFHF','HGFH', 'FFHF']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 9 #Enter the Goal state
P = env.env.P

def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    # Write your code here
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob, next_state, reward, done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi

# Finding the optimal policy
V_best_v, pi_best_v = value_iteration(P, gamma=0.99)

# Printing the policy
print("Name: CHANDRAPRIYADHARSHINI C")
print("Register number: 212223240019")
print()
print('Optimal policy and state-value function (VI):')
print_policy(pi_best_v, P)

# printing the success rate and the mean return
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_best_v, goal_state=goal_state)*100,
    mean_return(env, pi_best_v)))

# printing the state value function
print_state_value_function(V_best_v, P, prec=4)
```

## OUTPUT
Print the policy
<img width="797" height="384" alt="image" src="https://github.com/user-attachments/assets/49651118-88da-4f7a-9852-3a804bfaf485" />

Print the state value function
<img width="770" height="203" alt="image" src="https://github.com/user-attachments/assets/79f886d2-18c8-4980-88da-77d074b04c15" />

## RESULT
Thus the program successfully executed.
