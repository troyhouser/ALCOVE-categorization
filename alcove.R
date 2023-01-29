calculate_distance = function(exemplar,hidden,alpha,r=1){
  total_distance = 0
  for(i in 1:length(exemplar)){
    dim_distance = alpha[i]*(abs(exemplar[i]-hidden[i])**r)
    total_distance = total_distance + dim_distance
  }
  final_distance = total_distance**(1/r)
  return(final_distance)
}

hidden_activation = function(exemplar,hidden,alpha,c,r=1,q=1){
  distance = rep(NA,nrow(hidden))
  for(node in 1:nrow(hidden)){
    distance[node] = calculate_distance(exemplar, hidden[node,], alpha, r)
  }
  hidden_a = exp(-c*(distance**q))
  return(hidden_a)
}

activation_output = function(W,hidden_a){
  output_a = hidden_a %*% W
  return(output_a)
}

output_probs = function(phi,output_a){
  output_p = exp(phi*output_a)/sum(exp(phi*output_a))
}

exemplars = matrix(as.numeric(unlist(trainset[,1:4])),8,4,byrow=T)
exemplars
labels = c(0,0,0,0,1,1,1,1)

alcove_list = list(c=1,
                   lambdaW=0.9,
                   lambdaAlpha=0.9,
                   phi=10,
                   W=matrix(c(1,0,0,1),2,2,byrow=T),
                   alpha=rep(0.01,4),
                   hiddennodes = matrix(c(1,0,
                                          0,0,
                                          0,1,
                                          1,1),2,4,byrow=T),
                   r=1,
                   q=1)
forward_pass = function(exemplar,alcove_list){
  hidden_a = hidden_activation(exemplar = exemplar,
                               hidden=alcove_list$hiddennodes,
                               alpha=alcove_list$alpha,
                               c=alcove_list$c,
                               r=alcove_list$r,
                               q=alcove_list$q)
  output_a = activation_output(W=alcove_list$W,
                               hidden_a)
  output_p = output_probs(alcove_list$phi,
                          output_a)
  forward_output = list(hidden_activation=hidden_a,
                        activation_output=output_a,
                        output_probs=output_p)
  return(forward_output)
}

humble_teacher = function(activation_output,feedback){
  teacher = rep(NA,length(feedback))
  for(k in 1:length(feedback)){
    if(feedback[k]==1){
      teacher[k] = max(c(1,activation_output[k]))
    }else{
      teacher[k] = min(c(-1,activation_output[k]))
    }
  }
  return(teacher)
}

error = function(teacher, activation_output){
  e = 0.5*sum((teacher-c(activation_output))**2)
  return(e)
}

deltaW = function(lambdaW,teacher,activation_output,hidden_activation){
  hidden_activation = matrix(hidden_activation,ncol=1)
  output_error = matrix(teacher-c(activation_output),
                        nrow=1)
  W_deltas = lambdaW*(hidden_activation%*%output_error)
  return(W_deltas)
}
deltaAlpha = function(lambdaA,teacher,activation_output,hidden_activation,
                  W,hidden,c,exemplar){
  output_error = matrix(teacher-c(activation_output),
                        nrow=1,ncol=2)
  W_transpose = t(W)
  weighted_errors = output_error %*% W_transpose
  alpha_deltas = rep(0,length(exemplar))
  
  for(node in 1:nrow(hidden)){
    alpha_deltas = alpha_deltas+weighted_errors[1,node]*hidden_activation[node]*
      c*abs(hidden[node,]-exemplar)
  }
  alpha_deltas = -lambdaA*alpha_deltas
  return(alpha_deltas)
}

deltaW(lambdaW = alcove_list$lambdaW, teacher = c(1,-1),
        activation_output = forward_output$activation_output, 
        hidden_activation = forward_output$hidden_activation)

deltaAlpha(lambdaA = alcove_list$lambdaA, teacher = c(1,-1), 
           activation_output = forward_output$activation_output,
            hidden_activation = forward_output$activation_hidden, W = alcove_list$W, 
            hidden = alcove_list$hiddennodes, c = alcove_list$c, exemplar = exemplars[1,])

backward_pass = function(exemplar,alcove_list,label,forward_output){
  trial_teacher = humble_teacher(forward_output$activation_output,
                                 label)
  trial_deltaW = deltaW(lambdaW=alcove_list$lambdaW,teacher=trial_teacher,
                        activation_output=forward_output$activation_output,
                        hidden_activation=forward_output$hidden_activation)
  trial_deltaW = trial_deltaW[1:2,1:2]
  trial_deltaAlpha = deltaAlpha(lambdaA=alcove_list$lambdaAlpha,
                        teacher=trial_teacher,
                        activation_output=forward_output$activation_output,
                        hidden_activation=forward_output$hidden_activation,
                        W=alcove_list$W,
                        hidden=alcove_list$hiddennodes,
                        c=alcove_list$c,
                        exemplar=exemplar)
  alcove_list$W=alcove_list$W+trial_deltaW
  alcove_list$alpha=alcove_list$alpha+trial_deltaAlpha
  trial_error=error(trial_teacher,forward_output$activation_output)
  backward_list=list(alcove_list=alcove_list,trial_error=trial_error)
  return(backward_list)
}

ftest=forward_pass(exemplars[1,],alcove_list)
ftest
labels = cbind(labels,1-labels)
backward_pass(exemplars[1,],alcove_list,label=labels[1,],forward_output=ftest)


training = function(exemplars,alcove_list,labels,storeAlcove=F){
  output_error=rep(NA,nrow(exemplars))
  simulated_response=matrix(NA,nrow=nrow(labels),ncol=ncol(labels))
  if(storeAlcove){
    output_alcove_lists=rep(list(NA),nrow(exemplars))
  }
  for(i in 1:nrow(exemplars)){
    forward_trial=forward_pass(exemplars[i,],alcove_list)
    backward_trial=backward_pass(exemplars[i,],alcove_list,labels[i],
                                 forward_output=forward_trial)
    output_error[i]=backward_trial$trial_error
    simulated_response[i,]=forward_trial$output_probs
    if(storeAlcove){
      output_alcove_lists[[i]]=backward_trial$alcove_list
    }
    alcove_list=backward_trial$alcove_list
  }
  if(storeAlcove){
    return(list(output_alcove_lists=output_alcove_lists,
                output_error=output_error,
                resp_probs=simulated_response))
  }else{
    return(list(output_alcove_list=alcove_list,output_error=output_error,
                resp_probs=simulated_response))
  }
}
exemplars=matrix(c(1,1,1,1,
                   0,1,1,1,
                   1,1,0,0,
                   1,0,0,0,
                   1,0,1,0,
                   0,0,1,0,
                   0,1,0,1,
                   0,0,0,1),8,4,byrow=T)
training_test=training(exemplars=exemplars,alcove_list=alcove_list,labels=labels,
                       storeAlcove = T)
training_test
training_test$resp_probs[,2048]
