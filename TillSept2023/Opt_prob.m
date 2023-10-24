% Inputs
clear all
Ka=200;   %Number of active users
n=16;     %Number of blocks
J=16;     %Length of each coded block
M = 16*16; %Length of tree codeword
B = 128;   %Payload size
epsilon = 0.005; %Target error probability of tree decoder
cvx_begin gp
    variables p(n-1) % 1/2^paritylength vector
    q=1-p;
    
    %% Expected Complexity
    S=0;
    for i=1:n-2
        for m=1:i
            S=S+(Ka^(i-m))*(Ka-1)*prod(p(m:i));
        end
    end
    Expcomp = Ka*(1+S+n-2);
    %% Prob of Error Bound
    Bound = 0;
    for m=1:n-1
        Bound=Bound+(Ka^(n-1-m))*(Ka-1)*prod(p(m:n-1));
    end
    
    %% Optimization
    
    minimize(Expcomp)
    subject to
        Bound <= epsilon;
        prod(p) == 2^(-(M-B));
        p <= 1*ones(n-1,1);
        p >= (2^(-J))*ones(n-1,1);
cvx_end


% Round parity lengths to nearest integers
parity_length_real = log2(1./p);
disp(parity_length_real)
parity_length_integer = round(parity_length_real); % This might sometimes give a total parity length that does not sum to M-B. 
                                                   % If that is the case, adjust the vector to ensure sum is M-B.
                                                   % This step would result in an approximate solution, which is sufficient for practical purposes.
                                                   
                                                   
                                                  
disp(parity_length_integer)

