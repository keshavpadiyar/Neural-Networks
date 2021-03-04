%% Initialization
%  Initialize the world, Q-table, and hyperparameters
world = 4;
gwinit(world); % initialise the world environmnet

wState = gwstate(); % get the initial state

actions = [1 2 3 4]; % 1=down, 2=up, 3=right and 4=left

Q = zeros(wState.ysize,wState.xsize, length(actions));% initializing the Q table

% Forcing Q values at the borders to -inf so that bot wont get stuck there
Q(1, :, 2) = -Inf; % top 
Q(:, end, 3) = -Inf; % right 
Q(end, :, 1) = -Inf; % bottom 
Q(:, 1, 4) = -Inf; % left

eta = 0.5; % learning rate

gamma = 0.9; % discount factor

actionProb = ones(1,length(actions))/length(actions); %probabilities for each action

maxEpisodes = 10000;

%% Training loop
%  Train the agent using the Q-learning algorithm.

for episode = 1:maxEpisodes
    
    % Prining values for debugging
    if ~rem(episode,500)
        disp(episode);
    end
    
    % beginnig Q-Learning
    % Re-initialize the environment and states
    gwinit(world); % initialise the world environmnet
    wState = gwstate(); % get the initial state
    
    % execute every epesode till it reaches some terminal position
    while ~wState.isterminal
        [currentAction, optimalAction] = chooseaction(Q, wState.pos(1), wState.pos(2), actions, actionProb, getepsilon(episode,maxEpisodes));
        nextWState = gwaction(currentAction);
            if nextWState.isvalid
                reward = nextWState.feedback;
                Q(wState.pos(1),wState.pos(2),currentAction) = (1-eta) * Q(wState.pos(1),wState.pos(2),currentAction)...
                    + eta * (reward+ gamma * max(Q(nextWState.pos(1),nextWState.pos(2),:)));
                wState = nextWState;
            else
                break;
                
            end
    end 
    
end

figure(1);
gwdraw(); 
for x = 1:wState.ysize
   for y = 1:wState.xsize
      [~,I] = max(Q(x,y,:));
      gwplotarrow([x,y],I);
   end
end
%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.