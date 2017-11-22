
clear;
net=vgg16();
layer='fc6';
testFeatures=zeros(60,4096);
testFeaturesB=zeros(60,4096);
j=1;
for i=1:2:30
disp('i=');
disp(i);
add='/home/kishan/Desktop/Aditya/Snippet';
A=VideoReader(strcat(add,num2str(i),'.','mp4','.avi'));
B=VideoReader(strcat(add,num2str(i+1),'.','mp4','.avi'));
testFeatures=zeros(60,4096);
testFeaturesB=zeros(60,4096);
for k=1:60
   disp('k=');
   disp(k);
   this_frame=read(A,k);
   this_frameB=read(B,k);
   frame=imresize(this_frame,[224 224]);
   frameB=imresize(this_frameB,[224 224]);
 %  display("frameread");
   testFeatures(k,:) = activations(net,frame,layer);
   testFeaturesB(k,:) = activations(net,frameB,layer);

   % display("testfeatures");
   
   %{
   
thisfig = figure();
thisax = axes('Parent', thisfig);
image(this_frame, 'Parent', thisax);
title(thisax, sprintf('Frame #%d', k));
%}
end
%display("finished calculating");
testFeatures=testFeatures';
testFeaturesB=testFeaturesB';
i=1;
for k=1:10:60
    disp('kk=');
    disp(k);
    pooled(i,:)=max(testFeatures(:,k:k+9)');
    pooledB(i,:)=max(testFeaturesB(:,k:k+9)');
    i=i+1;
%display(k);
end
total_pooled=[pooled pooledB];
trainExamples(j,:) =total_pooled(:)';
j=j+1;
end
save data.mat trainExamples