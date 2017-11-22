clear;
re=load('data.mat');
re=struct2table(re);
trainExamples=table2array(re);




nInput = 12*4096; % number of nodes in input
nOutput = 2; % number of nodes in output
nHiddenLayer1 = 512; % number of nodes in th hidden layer
nHiddenLayer2 = 512;
nTrain = 15; % size of training set
alpha = 0.001; % learning rate

trainExamples = [ones(1,nTrain)',trainExamples];

w1 = rand(nInput+1,nHiddenLayer1);
w2=  rand(nHiddenLayer1+1,nHiddenLayer2);
w3 = rand(nHiddenLayer2+1,nOutput);

m=4;
rho=0.99;
v=zeros(((nInput+1)*nHiddenLayer1)+((nHiddenLayer1+1)*nHiddenLayer2)+((nHiddenLayer2+1)*nOutput),1);
S1(nHiddenLayer1+1,nTrain) = 0;
Se(nHiddenLayer2+1,nTrain) = 0;

S2(nOutput,nTrain) = 0;
S(nHiddenLayer1+1,nTrain) = 0;

%Y(1,:) = sign(trainExamples(2,:).^2-4*trainExamples(1,:).*trainExamples(3,:));
%Y(2,:) = -1*sign(trainExamples(2,:).^2-4*trainExamples(1,:).*trainExamples(3,:));
Y(1,:)=[1,0,1,0,1,1,0,1,0,1,1,1,1,0,1];
Y(2,:)=[0,1,0,1,0,0,1,0,1,0,0,0,0,1,0];
error1=[];
error2=[];
for i=1:50
    disp('i=');
    disp(i);
    for j=1:nTrain
       
        x = (trainExamples(j,:))';
        S(2:end,j)=w1'*x;

        S1(:,j)=max(S(:,j),0);
        S1(1,j)=1;
        
        S(2:end,j)=w2'*S1(:,j);
        Se(:,j)=max(S(:,j),0);
        Se(1,j)=1;
        
        S2(:,j)=w3'*Se(:,j);
        
        E2=S2(:,j)-Y(:,j);
        delta2=Se(:,j)*E2';
        delta2=delta2./m;
        at=E2';
       error1(i)=at(1);
       error2(i)=at(2);
        E=w3*E2;
        E4=max(E,0);
        E1=E4(2:end,:);
        delta1=S1(:,j)*E1';
        delta1=delta1./m;
        
        E=w2*E1;
        E4=max(E,0);
        E1=E4(2:end,:);
        delta0=x*E1';
        delta0=delta0./m;
       
       p= (nHiddenLayer1 * (nInput + 1));
        delta=[delta0(:);delta1(:);delta2(:)];
        w=[w1(:);w2(:);w3(:)];
        v=rho*v+delta;
        w=w-alpha*v;
        
        w1 = reshape(w(1:nHiddenLayer1 * (nInput + 1)), ...
                  (nInput + 1),nHiddenLayer1);
        w2 = reshape(w((nHiddenLayer1 * (nInput + 1))+1:(nHiddenLayer2 * (nHiddenLayer1 + 1))+p), ...
                  (nHiddenLayer1 + 1),nHiddenLayer2);

        w3 = reshape(w((1 + p+(nHiddenLayer2 * (nHiddenLayer2 + 1))):end), ...
                  (nHiddenLayer2 + 1),nOutput);
        
    end
   %{
 k=0;
    for j=1:10
    k=k+sum(sum((-Y).*log(S2(:,j)) - (1-Y).*log(1-(S2(:,j))), 2))/m;
        
    end
    re(i)=abs(k);
 %}   
    
   
end
%plot([1:10],re);
