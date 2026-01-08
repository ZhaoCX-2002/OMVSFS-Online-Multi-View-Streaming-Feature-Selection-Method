function [ selectedFeatures,time ] = OSGFS_FI_inter_clean(X,Y,G,alpha)
start=tic;
[~,P]=size(X);   % 获取数据维度：P为特征总数
mode=zeros(1,P); % 初始化特征选择标记（0未选，1选中）
for i=1:G:P      % 按组大小G遍历特征
    i_end=G+i-1; % 当前组结束索引
    if i_end>P   % 处理最后一组不足G的情况
        i_end=P;
    end
    indexArray=[find(mode==1),i:i_end];% 合并已选特征和当前组特征
    X_G=X(:,indexArray);% 提取当前组和之前已选特征的数据
    mode_G=OSGFS_FI_intra(X_G,Y);% 调用组内选择函数
    G_N=length(mode_G); % 当前组选中的特征数
    for j=1:G_N
          ind=mode_G(1,j); % 获取选中特征的相对索引
          index=indexArray(ind);% 转换为全局索引
          mode(1,index)=1;% 标记为选中
    end
end
intraSelectedFeatures=find(mode==1);
X_Inter=X(:,intraSelectedFeatures);% 提取选中特征的数据
[B,FitInfo]=lasso(X_Inter,Y,'Alpha',alpha,'CV',5);% 弹性网络回归
minInd=FitInfo.IndexMinMSE;% 选择最小MSE对应的系数
SF=B(:,minInd);
selectedFeatures=intraSelectedFeatures(SF~=0);% 非零系数对应的特征
time=toc(start);   % 计算总时间
end