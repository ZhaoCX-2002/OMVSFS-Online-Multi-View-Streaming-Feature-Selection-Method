function   [selected_features, time]=osfs_d(data1,class_index,alpha,test)
% for continue value
% 确保所有值从1开始
%data1(:,1:class_index-1) = data1(:,1:class_index-1) - min(data1(:,1:class_index-1)) + 1;
features = data1(:, 1:class_index-1);  % 前 p-1 列是特征
labels = data1(:, class_index);        % 最后一列是标签
min_feat = min(features(:));   % 计算特征最小值
    if min_feat < 1
        features = features - min_feat + 1;
    end
        data1 = [features, labels];
%data1 = data1 - min(data1(:)) + 1;
[n,p]=size(data1);
ns=max(data1);
selected_features=[];
selected_features1=[];
b=[];

start=tic;

 for i=1:p-1
     
      
     %for very sparse data 
     n1=sum(data1(:,i));
      if n1==0
        continue;
      end     
     
         
     stop=0;
     CI=1;
        
     [CI] = my_cond_indep_chisquare(data1,i, class_index, [], test, alpha, ns);
      
      if CI==0
          stop=1;
          selected_features=[selected_features,i];
      end
         
      if stop
          
          p2=length(selected_features);
          selected_features1=selected_features;
          
           for j=1:p2
               
              b=setdiff(selected_features1, selected_features(j),'stable');
               
               if ~isempty(b)
                  [CI]=compter_dep_2(b,selected_features(j),class_index,3, 1, alpha, test,data1);
      
                   if CI==1
                      selected_features1=b;
                   end
              end
          end
     end   
   selected_features=selected_features1;
 end
 
  time=toc(start);
  
    
      