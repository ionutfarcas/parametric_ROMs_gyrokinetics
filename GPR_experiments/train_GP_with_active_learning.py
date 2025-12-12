import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def read_GENE_data(file):

    growth_rate = []
    frequency   = []

    with open(file) as file:

        lines = file.readlines()
        for i, line in enumerate(lines):
            if i >= 1:
                tokens = line.split()
                
                growth_rate.append(np.float64(tokens[-2]))
                frequency.append(np.float64(tokens[-1]))

    file.close()

    return np.array(growth_rate), np.array(frequency)

def read_GENE_parameters(file, n_points, dim):

    params = np.zeros((n_points, dim))

    with open(file) as file:

        lines = file.readlines()
        for i, line in enumerate(lines):
            if i >= 1:
                tokens = line.split()
                
                params[i-1, 0] = np.float64(tokens[1])
                params[i-1, 1] = np.float64(tokens[2])
                params[i-1, 2] = np.float64(tokens[3])
                params[i-1, 3] = np.float64(tokens[4])
                params[i-1, 4] = np.float64(tokens[5])
                params[i-1, 5] = np.float64(tokens[6])

    file.close()

    return params

dim          = 6 # number of parameters
n_points     = 1000 # number of testing parameters 

file = './data/data_stellarator_case_mode_transitions.txt'

params      = read_GENE_parameters(file, n_points, dim)
gr, freq    = read_GENE_data(file)

# mask = np.abs(gr) >= 1e-5

# params  = params[mask, :]
# gr      = gr[mask]

n_test = 200
n_pool = gr.shape[0] - n_test

X_pool = params[n_test:, :]
y_pool = gr[n_test:].reshape(-1, 1)


X_test = params[:n_test, :]
y_test = gr[:n_test]

def select_max_variance(X_pool, gp):

    mu, sigma = gp.predict(X_pool, return_std=True)
    
    return np.argmax(sigma)

initial_size    = 10
idx_init        = np.random.choice(len(X_pool), size=initial_size, replace=False)

X_train = X_pool[idx_init]
y_train = y_pool[idx_init]

# Remove initial points from pool
mask            = np.ones(len(X_pool), dtype=bool)
mask[idx_init]  = False
X_pool          = X_pool[mask]
y_pool          = y_pool[mask]

# -----------------------------------------------------
# Build GP model
# -----------------------------------------------------
# kernel =  ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-3, 1e3), 
#                                       nu=0.5) + WhiteKernel(1e-3, (1e-4, 1e0))

# kernel = 0.5 * Matern(nu=1.5) + 0.5 * RBF(length_scale=2.0)
kernel = 0.5 * Matern(nu=0.5) + 0.5 * RBF(length_scale=2.0)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-8,
    normalize_y=False,
    n_restarts_optimizer=5
)

# -----------------------------------------------------
# 5. Active learning loop (select from X_pool, y_pool)
# -----------------------------------------------------
n_iterations = 100
for it in range(n_iterations):

    # Fit GP on current training set
    gp.fit(X_train, y_train)

    best_idx = select_max_variance(X_pool, gp)
    ## best_idx = select_ei(X_pool, gp, y_train)  # expected improvement


    x_next = X_pool[best_idx].reshape(1, -1)
    y_next = y_pool[best_idx].reshape(1, -1)

    # Add to training data
    X_train = np.vstack([X_train, x_next])
    y_train = np.vstack([y_train, y_next])

    # Remove from pool
    X_pool = np.delete(X_pool, best_idx, axis=0)
    y_pool = np.delete(y_pool, best_idx, axis=0)

    print(f"Iteration {it+1:2d}: selected y = {float(y_next):.4f}, remaining pool = {len(X_pool)}")

print("Active learning complete.")

print(f"Final training size: {len(X_train)}")
print(f"Remaining test size:   {len(X_pool)}")

# -----------------------------------------------------
# 6. Evaluation on remaining pool (X_test, y_test)
# -----------------------------------------------------
# Final GP fit with all selected AL points
gp.fit(X_train, y_train)

# Predict on test set
y_pred_mean, y_pred_std = gp.predict(X_test, return_std=True)

print("Predictions")
errors = np.zeros(n_test)
for i in range(n_test):

    errors[i] = np.abs(y_test[i] - y_pred_mean[i])/np.abs(y_test[i])

    print(f"Predicted Mean: {y_pred_mean[i]:.4f} | Std Dev: {y_pred_std[i]:.4f} | Ref value: {y_test[i]:.4f} | Rel error: {np.abs(y_test[i]-y_pred_mean[i])/np.abs(y_test[i]):.4f}")

print(f"Min error: {np.min(errors):.4f} | Max error: {np.max(errors):.4f} | Mean error: {np.mean(errors):.4f}")

# Metrics
mae     = mean_absolute_error(y_test, y_pred_mean)
rmse    = np.sqrt(mean_squared_error(y_test, y_pred_mean))
r2      = r2_score(y_test, y_pred_mean)

print("\nTest performance on remaining pool:")
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R²   = {r2:.4f}")

# np.save('./data/freq_ref.npy', y_test)
# np.save('./data/freq_GP_pred_mean.npy', y_pred_mean)
# np.save('./data/freq_GP_pred_std.npy', y_pred_std)

# np.save('./data/gr_ref.npy', y_test)
# np.save('./data/gr_GP_pred_mean.npy', y_pred_mean)
# np.save('./data/gr_GP_pred_std.npy', y_pred_std)
