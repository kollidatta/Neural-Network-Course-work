%%% Best accurate results are 13% 
%%Epoch used : 20
%%Architecture : 64 32 8 1
%%Submitted By : Sridatta kolli



function er=finance(ppp,c)
  load('Finance_a');
  n=size(ppp,2)-1;
  
  % Initializing.
  X=cell(1, n); W=cell(1, n); B=cell(1, n);  Y=cell(1, n);
  DX=cell(1, n); DDX=cell(1, n);             
  for i=1:n
      W{i}=0.5*ones(ppp(i+1),ppp(i)) - rand(ppp(i+1),ppp(i));
      B{i}=0.5*ones(ppp(i+1),1) - rand(ppp(i+1),1); 
      X{i}=zeros(ppp(i),1);      
      Y{i}=zeros(ppp(i+1),1);
      DX{i+1}=zeros(ppp(i+1),1);
      DDX{i+1}=zeros(ppp(i+1),ppp(i+1));     
  end
for k = 1:20  
  % The Training 
  N=690; M=345;
  sp=randsample(N,M+M); 
  alpha=0.01;
  for i=1:M 
      % Computing the output
      X{1}=x(1:15,sp(i));%3TO 5      
      for j=1:n-1  % The first n-1 layers
          Y{j}=B{j} + W{j}*X{j}; 
         DX{j+1}=Y{j}>0;
          X{j+1}=DX{j+1}.*Y{j};
          DDX{j+1}=diag(DX{j+1});
      end
      Y{n}=B{n} + W{n}*X{n};  % Contineous output from the last layer
      if(c==1)
          Y{n}=Y{n}>0;     % Binary output
      end                
      e = x(16,sp(i)) - Y{n};
      
      % Back Propagation
      DB = alpha*e;
      B{n}=B{n} + DB;
      W{n}=W{n} + DB*X{n}';
      for j=n-1:-1:1
          DB = (W{j+1}*DDX{j+1})'*DB;          
          B{j} = B{j} + DB;
          W{j} = W{j} + DB*X{j}';  
      end       
  end
  
% Testing
er=0;
for i=M+1:M+M
    
      X{1}=x(1:15,sp(i));      
      for j=1:n-1  % The first n-1 layers
          Y{j}=B{j} + W{j}*X{j}; 
         DX{j+1}=Y{j}>0;
          X{j+1}=DX{j+1}.*Y{j};
          DDX{j+1}=diag(DX{j+1});
      end
    Y{n}=B{n} + W{n}*X{n};
    if(c==1)
        Y{n}=Y{n}>0;
    end                  
    e = abs(x(16,sp(i)) - Y{n});
    if(c==0)
       e=e/x(16,sp(i));  % Relative error
    end
    er = er + e;
end

er=er/M;
a(k)=er;
end         

disp('min ave error');
min(a)
return

     
