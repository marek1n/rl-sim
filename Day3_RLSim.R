
#####
#
# Simulating an individual reinforcement-learning experiment (Mind & Brain, day 3)
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


# for utilities
library(RColorBrewer)
library(scales)


#Simulation function
#First we define the simulation function with a set of parameters

Sim_fct <- function(Tmax = 100,
                    N,              # Number of individuals
                    N_options = 2, 
                    Payoff_Better,
                    Payoff_Worse, 
                    Payoff_SD,
                    phi,            # learning rate
                    lambda          # exploration/exploitation rate
                    ) {
  
  Q_values <- matrix(0.1, nrow = N, ncol = N_options )
  
  #Create output object to record choices and payoffs participants received as well as observed choices and latent Q_values
  Result <- list(id=rep(1:N, each = Tmax),
                  trial= rep(1:Tmax, N), 
                  Choice=NA, 
                  Correct=NA, 
                  Payoff=NA, 
                  Q_values = array(NA, dim = c(Tmax, N, N_options))
                  )
  
  # Start simulation loop
  for (trial in 1:Tmax) {
    #Store Q values
    Result$Q_values[trial,,] <- Q_values
    
    #Loop over all individuals
    for (id in 1:N){
      #Individual choice probabilities
      p <- sapply(1:N_options, function(option)  exp(lambda * Q_values[id, option]) / sum (exp(lambda * Q_values[id,]) ))
      #Make choice proportional to values
      choice <- sample(1:N_options, size = 1, prob = p)
      
      #Generate a payoff based on choice
      
      #Individual chose better option
      if ( (trial <= Tmax/2 & choice == 1) | (trial > Tmax/2 & choice == 2)  ) {
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
      
      #Updating all options (our default here)
      Q_values[id,] <- (1-phi)  *Q_values[id,] + phi * pay
      
      #Updating only chosen option
      #Q_values[id,choice] <- (1-phi)  *Q_values[id,choice] + phi * pay[choice]
      
    }#individual id
    
  }#trial
  
  #Record parameter values
  Result$phi = phi
  Result$lambda = lambda

  
  return(Result)
  
}#sim_funct

#Run simulation

Result <- Sim_fct(N=1,  #Plot works best for 1-5 individuals
                  
                  #Choice environment
                  Payoff_Better=0.6,       # > 0
                  Payoff_Worse=0.59,        # >= 0
                  Payoff_SD=0,           # >= 0
                  
                  #Individual learning parameters
                  phi=0.1,  #Learning rate; 0 <-> 1
                  lambda=2) #Exploration/Exploitation rate; >= 0
                  

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
  arrows(49.5,1.7,49.5,1.3, length = 0.1, lwd = 2)  
}



####
###
##
# TASKS
##
###
####

#We start with a very simple scenario with binary rewards (0 vs.1) and no noise:
#1. Vary phi and describe its effect on learning and behavior?
#2. Vary lambda and describe its effect on learning and behavior?
#3. Which combination(s) of both learning parameters lead to best learning outcomes and why?
  
#Now set Payoff_Better=15 and Payoff_Worse=12:
#4. Vary payoff stochasticity (which determines difficulty). Describe its effect on learning and "optimal" parameter combinations. 
#Note that the effect of lambda depends on payoff values, so lambda values between 0 and 0.5 are more sensible now.

#5. Does is change dynamics if we only update chosen option?




