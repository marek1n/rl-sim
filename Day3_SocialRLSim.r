
#####
#
# Simulating a social-learning experiment (Mind & Brain, day 3)
#
####


#We assume that there is always one option with higher expected reward compared to the rest
#The optimal option switches after half of the trials

# Individual learning
# Use reinforcement recursion:
# A_i,t = (1-phi)*A_i,t-1 + phi*P_i,t
# where A_i,t is latent attraction (value) to option i at time t
# phi is updating parameter
# P_i is payoff from option i at time t

# Individual choice probability given by softmax
# P_ind = exp( lambda*A_1,t ) / Sum( exp(lambda*A_t) )
# where lambda is a scaling parameter to add noise (i.e., exploration) to choice.
# As lambda -> 0, choice becomes noisy.

#Social learning
# Social choice probability
# P_social = n_i^theta / Sum(n^theta)
# where n_i is the number of individuals who chose option i in the previous round
# and theta is the conformity exponent which controls the strength of conformist learning

# Combination of asocial and social information
# P = (1-sigma)*P_ind + sigma * P_social



# for utilities

library(RColorBrewer)
library(scales)


#Simulation function
#First we define the simulation function with a set of parameters

Sim_fct <- function(Tmax = 100, Group_size, N_options = 2, Payoff_Better,Payoff_Worse, Payoff_SD, phi,lambda, sigma, theta){
  
  Q_values <- matrix(0.1, nrow = Group_size, ncol = N_options )
  
  #Create output matrix to record choices and payoffs participants received as well as observed choices and latent Q_values
  
  Result <- list(id=rep(1:Group_size, each  = Tmax),trial= rep(1:Tmax, Group_size), Choice=NA, Correct=NA, Payoff=NA, Q_values = array(NA, dim = c(Tmax, Group_size, N_options))
  )
  
  # Start simulation loop
  for (trial in 1:Tmax) {
    
    Result$Q_values[trial,,] <- Q_values
    
    #Generate choices
    
    #Loop over all individuals
    for (id in 1:Group_size){
      
      #Individual choice probabilities
      p_individual <- sapply(1:N_options, function(option)  exp(lambda * Q_values[id, option]) / sum (exp(lambda * Q_values[id,]) ))
      
      #1st round or solo participants
      #There are no choices to copy, so only individual information
      if(trial==1 | Group_size == 1){
        p <- p_individual 
      } else {
        #From 2nd round, combination of individual and social information
        
        #Frequency all options were chosen by group members
        n <-  sapply(1:N_options, function(option) length(which(Result$Choice[Result$trial==trial-1 & Result$id != id] == option)) )
        
        #Compute probability for each option depending on weight of social learning learning strategy (conformity exponent)
        p_social <- sapply(1:N_options, function(option) n[option]^theta / sum(n^theta) )
        
        #Compute choice probabilities based on individual and social probabilities
        p <- (1-sigma) * p_individual + sigma * p_social 
        
      }
      
      
      #Make choice proportional to attraction scores and social information
      choice <- sample(1:N_options, size = 1, prob = p)
      
      #Generate a payoff based on choice
      
      #Individual chose better option
      if ( (trial <= Tmax/2 & choice == 1) | (trial > Tmax/2 & choice == 2)  ){
        payoff <- round(rnorm(1, mean=Payoff_Better, sd=Payoff_SD))
        while (payoff<0) payoff <- round(rnorm(1, mean=Payoff_Better, sd=Payoff_SD))
        correct <- 1
        
        #Individual chose worse option
      } else {
        payoff <- round(rnorm(1, mean=Payoff_Worse, sd=Payoff_SD))
        while (payoff<0) payoff <- round(rnorm(1, mean=Payoff_Worse, sd=Payoff_SD))
        correct <- 0
      }
      
      #Record choice & Payoff
      Result$Choice[which(Result$id==id & Result$trial==trial)]  <- choice
      Result$Payoff[which(Result$id==id & Result$trial==trial)]  <- payoff
      Result$Correct[which(Result$id==id & Result$trial==trial)] <- correct
      
      #Update attraction scores based on payoff
      
      pay <- rep(0, N_options)
      pay[choice] <- payoff
      Q_values[id,] <- (1-phi)  *Q_values[id,] + phi * pay
      
    }#individual id
    
  }#trial
  
  #Record parameter values
  Result$phi = phi
  Result$lambda = lambda
  Result$sigma = sigma
  Result$theta = theta
  
  
  return(Result)
  
}#sim_funct

#Run simulation

Result <- Sim_fct(Group_size=5,  #Plot works best for 1-5 group members
                  
                  #Choice environment
                  Payoff_Better=15,
                  Payoff_Worse=12,
                  Payoff_SD=1,
                  
                  #Individual learning parameters
                  
                  phi=0.4,     #Learning rate
                  lambda=0.2, #Exploration/Exploitation rate
                  
                  #Social learning parameters
                  
                  sigma=0.3, #Social learning weight
                  theta=2) #Conformity Exponent


#Plot

{
  
  par(mar = c(2,3,0,0), oma = c(0,0,2,0))
  
  #color stuff
  x <- seq(from=0, to=1, by=0.2) # fake data
  col.pal <- brewer.pal(length(x), "Dark2") #create a pallette
  
  
  plot(Result$Q_values[, 1, 1], type  = "n", ylim = c(-1.3, 1.3), xlab = "",ylab = "", bty = "n", xaxt = "n", yaxt = "n")
  axis(side = 2, labels = FALSE, tck = 0, lwd = 2)
  abline(v = 49.5, lty  =1, col = "grey")
  abline(h = 0, lty  =2, col = "grey")
  
  rect(-3, -1.35, 49.5, 0, density = NA, col = alpha("grey", alpha = 0.3), border = NULL)
  
  rect(49.5, 0, 101, 1.35, density = NA, col = alpha("grey", alpha = 0.3), border = NULL)
  
  
  for (id in 1:max(Result$id)) {
    color = col.pal[id]
    
    lines((Result$Q_values[, id, 2]-Result$Q_values[, id, 1])/max(Result$Q_values), col = color, lwd = 2)
    points(ifelse(Result$Choice[Result$id==id]==2, 1.1+(id*0.05), -1.1-(id*0.05)), col = color, pch = 16)
  }
  
  mtext("Switch in better option (from B to A)", side = 3, outer = FALSE, line = 0.5, cex = 1)
  mtext("Belief about relative payoffs", side = 2, outer = FALSE, line = 1, cex = 1.3)
  mtext("Trial", side = 1, outer = FALSE, line = 1, cex = 1.3)
  
  text("Option B", x = 4, y = -1.05)
  text("Option A", x = 4, y = 1.05)
  
  arrows(49.5,1.8,49.5,1.3, length = 0.1, lwd = 2)  
  
}




####
###
##
# TASKS
##
###
####

#Now, we keep the individual learning parameters constant and focus on social learning
#1. Vary sigma and describe its effect on collective learning and behavior?
#2. Vary theta and describe its effect on collective learning and behavior?
#3. How do both parameters interact and which combination(s) lead to best outcomes?
#4. How do socially learning groups compare to individual learners?
#5. Explore how social learning parameters interact with individual learning parameters.
