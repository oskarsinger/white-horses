## To mimic hearsteps, let the number of individuals be 37 and number of time points be 7770 (i.e., 37 users, 42 days, 5 decisions per day)

## How to generate an individual's baseline covaraites
age = rpois(1,35)		## participant's age, male = 1
gender = rbinom(1,1,.351)		## participant's gender
activity_base = rnorm(1, mean = 0, sd = 2.3)			## baseline acitvity level, mean centered, might need to change to zero inflated thing
activity_base_sq = activity_base^2
activity_age = activity_base*age

baseline = c(age, gender, activity_base, activity_base_sq, activity_age)

## at a given time t, you generate the following states
pre_steps = rnorm(1, mean = 2.7, sd = 3.03)		## previous step count
engaged_level = runif(1)			## measure of engagement
state_vec = c(pre_steps, engaged_level)

## coefficients for mean model- these were estimated from heartsteps
coeff_vec = c(1.737, 0.277, 0.335, -0.001, 0.281, -0.084, -0.034, 0.004, 0.064, 0.198, -0.919, 0.007, 0.251, 0.064, 0.032, -0.002, 0.004, -0.008, 0.027, 0.005, -0.002, -0.007, -0.574, 0.134, 0.027, 0.002, -0.005, -0.011, 0.018, -0.004, 0, 0.01, 0.136, -0.192, -0.053, 0.003)
 
 
# Generate reward for action = 1, i.e. suggestion
action = 1
covariate_vec = c(1, state_vec, baseline, action, action*state_vec, action*baseline, state_vec[1]*baseline,
state_vec[2]*baseline, state_vec[1]*baseline*action, state_vec[2]*baseline*action)
mean_1 = (covariate_vec%*%coeff_vec)[1,1]
reward_1 = rnorm(1, mean_1, sd = sqrt(7.43))


# Generate reward for action = 0 i.e. no suggestion
action = 0
covariate_vec = c(1, state_vec, baseline, action, action*state_vec, action*baseline, state_vec[1]*baseline,
state_vec[2]*baseline, state_vec[1]*baseline*action, state_vec[2]*baseline*action)
mean_0 = (covariate_vec%*%coeff_vec)[1,1]
reward_0 = rnorm(1, mean_0, sd = sqrt(7.43))


## for printing for latex

alpha_mat = c(coeff_vec[1], coeff_vec[4:8])

beta_mat1 = c(coeff_vec[2], coeff_vec[17:21])
beta_mat2 = c(coeff_vec[3], coeff_vec[22:26])
beta_mat3 = c(coeff_vec[9], coeff_vec[12:16])
beta_mat4 = c(coeff_vec[10], coeff_vec[27:31])
beta_mat5 = c(coeff_vec[11], coeff_vec[32:36])

beta_mat = cbind(beta_mat1, beta_mat2, beta_mat3, beta_mat4, beta_mat5)
paste(beta_mat[1,], collapse = ' & ')
paste(beta_mat[2,], collapse = ' & ')
paste(beta_mat[3,], collapse = ' & ')
paste(beta_mat[4,], collapse = ' & ')
paste(beta_mat[5,], collapse = ' & ')
paste(beta_mat[6,], collapse = ' & ')

paste(alpha_mat, collapse = "\\")