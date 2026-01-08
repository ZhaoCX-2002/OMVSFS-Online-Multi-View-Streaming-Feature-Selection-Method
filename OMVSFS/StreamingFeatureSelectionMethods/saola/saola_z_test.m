

function [current_feature,time] = saola_z_test(data,alpha)
%input  alpha=0.01/0.05
%Performs the SAOLA algorithm using mutual information measure by Yu 2014.
%data: columns denote features (attributes), while rows represent data instances.
%the last column of a data set is the class attribute
%output
%current_feature: selected features
%time: running time
start_time=tic;
[n,numFeatures] = size(data);%返回N样本例子；特征数量
class_a=numFeatures;%the index of the class attribute
current_feature=[];
dep=sparse(1,numFeatures-1);% 存储每个特征与类别的依赖度（稀疏矩阵）
CI=1;% 条件独立性标志（1=独立，0=相关）
for i = 1:numFeatures-1
    %for very sparse data 
     n1=sum(data(:,i));% 检查特征i是否全零（针对稀疏数据）
     if n1==0
        continue;
     end
     % 计算特征i与类别class_a的条件独立性
    [CI,dep(i)] = my_cond_indep_fisher_z(data,i, class_a, [],n,alpha);
    if CI==1 || isnan(dep(i))% 若独立或依赖度无效，跳过
        continue;
    end
       current_feature=[current_feature, i];   % 将特征i加入候选集合 
       %current_feature1=setdiff(current_feature,i,'stable');
        current_feature1=current_feature(~sum(bsxfun(@eq,current_feature',i),2));
        % 提取当前特征i在集合中的位置
    if ~isempty(current_feature1)      
          p=length(current_feature1);
          for j=1:p  
              % 检查特征i与已选特征current_feature1(j)的条件独立性
                 [CI, dep_ij] = my_cond_indep_fisher_z(data,i, current_feature1(j), [],n,alpha);                                              
                  if CI==1|| isnan(dep_ij)%不相关之间选
                     continue;
                  end
              % 相关了继续下面的路
                  t_dep=dep_ij;% 临时存储依赖度
                  t_feature=current_feature1(j);% 临时存储已选特征
                     % 依赖度比较与冗余判断
                   if dep(t_feature)>=dep(i) && t_dep>dep(i)%当前第j个特征与标签的偏相关系数、最新特征i与标签的系数；j与i俩个特征之间的相关系数、最新特征与标签的系数
                           %current_feature=setdiff(current_feature,i,
                           %'stable');%第一个公式说明已选特征t_feature与标签的相关性大于i的
                           %，第二个说明这俩个之间的相关性大于i对标签的相关性，说明。。。，因为他没有用条件下的求相关性，他用这种方法来使衡量他们特征间相关性和特征对标签相关性的一个衡量。
                           current_feature=current_feature(~sum(bsxfun(@eq,current_feature',i),2));
                           % 移除特征i  
                           break;
                   end
                  if dep(i)>dep(t_feature) && t_dep>dep(t_feature)
                       current_feature=current_feature(~sum(bsxfun(@eq,current_feature',t_feature),2));
                       % 移除已选特征t_feature
                  end   
          end   
    end
end
time=toc(start_time);


  
