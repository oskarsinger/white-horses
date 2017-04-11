## To mimic hearsteps, let the number of individuals be 37 and number of time points be 7770 (i.e., 37 users, 42 days, 5 decisions per day)


## How to generate an individual's baseline covaraites
age = rpois(1,35)		## participant's age
gender = rbinom(1,1,.351)		## participant's gender, male = 1
activity_base = rnorm(1, mean = 0, sd = 2.3)			## baseline acitvity level, mean centered, might need to change to zero inflated thing
baseline = c(age, gender, activity_base)

## at a given time t, you generate the following states
pre_steps = rnorm(1, mean = 2.7, sd = 3.03)		## previous step count
engaged_level = runif(1)			## measure of engagement
state_vec = c(pre_steps, engaged_level)

## coefficients for mean model- these were estimated from heartsteps
coeff_vec = c(1.462, 0.362, 0.3, 0.003, 0.231, -0.038, 0.206, 0.197, -1.126, 0.007, 0.368, 0.079, 0.002, -0.033, -0.02, -0.004, -0.354, 0.273, -0.006, -0.034, 0.001, 0.011, -0.091, -0.205)
 
 
# Generate reward for action = 1, i.e. suggestion
action = 1
covariate_vec = c(1, state_vec, baseline, action, action*state_vec, action*baseline, state_vec[1]*baseline,
state_vec[2]*baseline, state_vec[1]*baseline*action, state_vec[2]*baseline*action)
mean_1 = (covariate_vec%*%coeff_vec)[1,1]
reward_1 = rnorm(1, mean_1, sd = sqrt(7.47))


# Generate reward for action = 0 i.e. no suggestion
action = 0
covariate_vec = c(1, state_vec, baseline, action, action*state_vec, action*baseline, state_vec[1]*baseline,
state_vec[2]*baseline, state_vec[1]*baseline*action, state_vec[2]*baseline*action)
mean_0 = (covariate_vec%*%coeff_vec)[1,1]
reward_0 = rnorm(1, mean_0, sd = sqrt(7.47))

