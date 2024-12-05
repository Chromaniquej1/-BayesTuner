from bayes_opt import BayesianOptimization
import time

def bayesian_optimization(evaluate_network):
    # Bounded region of parameter space
    pbounds = {'dropout': (0.0, 0.499),
               'learning_rate': (0.0, 0.1),
               'neuronPct': (0.01, 1),
               'neuronShrink': (0.01, 1)}

    optimizer = BayesianOptimization(
        f=evaluate_network,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    start_time = time.time()
    optimizer.maximize(init_points=10, n_iter=30)
    time_took = time.time() - start_time

    print(f"Total runtime: {hms_string(time_took)}")
    print(optimizer.max)
