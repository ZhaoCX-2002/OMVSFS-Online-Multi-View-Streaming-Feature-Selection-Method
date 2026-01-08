function [ selectedFeatures,time ] = OSGFS_FI_intra(G,Y)
%OFS_INTERACTION 
% online streaming feature selection considering feature interaction
%
% Output:  selectedFeatures  选择的特征序号集合
%                time   算法运行时间
% Input:  X     样本属性数据矩阵
%            Y     样本标签矩阵
%            G     Group大小

start=tic;

[~,p]=size(G); % 获取组内特征数p
MI_Array=zeros(1,p);% 存储每个特征与Y的互信息（SU值）
S=ones(1,p);% 标记特征状态（1待选，-1排除，0已处理）
R=zeros(1,p);% 标记特征是否被选中（1选中，0未选）

for i=1:p
    MI_Array(1,i)=SU(G(:,i),Y);% 计算特征与标签的互信息（SU函数）
end
% disp(MI_Array);

I_R=0;% 记录当前最大互信息  首次选择特征时，任何与标签相关的特征都可能被选中。
while ~isempty(find(S==1, 1))             % 存在待选特征时循环；在矩阵S中寻找第一个值为1的元素的位置，只要矩阵S中还能找到值为1的元素，就继续执行循环
         [mi_val,I]=max(MI_Array);% 选择当前最大互信息的特征
         if mi_val<0 % 无有效特征时退出
             break;
         end
         
            current_index=I; % 当前处理特征索引
            S(1,current_index)=0; % 标记为已处理
            R(1,current_index)=1;% 标记为选中
            MI_Array(1,current_index)=-1; % 避免重复选择
%             disp(['--------current---',int2str(current_index)]);
            I_current=round(SU(G(:,R==1),Y)*10000)/10000;% 当前选中集的互信息
            unSelected=find(S==1); % 获取未选特征索引
            sum_unSelected=length(unSelected);
            interactArray=zeros(1,p);% 存储交互特征信息
            for j=1:sum_unSelected
                     unSelected_index=unSelected(1,j);
                     mi_inter=mi3(G(:,current_index),G(:,unSelected_index),Y);   % 三变量互信息 这个其实就是权重因子
                    if mi_inter >= 0 % 存在正向交互
                         S(1,unSelected_index)=-1;% 排除冗余特征
                         MI_Array(1,unSelected_index)=-1;
%                          disp(['-----negative index---',int2str(unSelected_index)]);
                     else% 负向交互需进一步判断
                          R(1,unSelected_index)=1;            % 临时选中  因为后面还要找最负的，就是协同性更大的         
                          MS_current=round(SU(G(:,R==1),Y)*10000)/10000;  
%                           disp(['-----interactive index---',int2str(unSelected_index)]);
%                           disp([mi_inter,MS_current,I_R]);
                          if MS_current>I_R 
                           interactArray(1,unSelected_index)=MS_current;% 记录交互增益
                           interactArray(1,unSelected_index)=mi_inter;
                          end    
                           R(1,unSelected_index)=0;              % 恢复未选状态           
                           S(1,unSelected_index)=-1; 
                           MI_Array(1,unSelected_index)=-1;
                    end                 
             end
             
            %没有选择交互特征
            if  any(interactArray<0)%~isempty(interactArray<0)查 interactArray 中是否存在负数元素。仅当 interactArray 初始化为空数组时，条件才不成立。这里其实用不着
                while ~isempty(find(interactArray<0, 1))%检查 interactArray 中是否存在至少一个负数元素。
                    [~,I]=min(interactArray);% 选择最负交互的特征 
                    R(1,I)=1; % 临时选中
                    mi_int=round(SU(G(:,R==1),Y)*10000)/10000;
                    if mi_int > I_R
                        I_R=mi_int;   % 更新最大互信息
                    else
                         R(1,I)=0;  % 取消选中
                    end 
                    interactArray(1,I)=0; % 标记已处理
                end
            else
%                 disp(['--------single---',int2str(current_index)])
%                 disp([I_current,I_R]);
                if I_current<=I_R% 当前特征无增益   //为什么组越小选择的特征越多原因就在这里，因为这里的代码意味着
                    S(1,current_index)=-1; % 排除   如果最大互信息的那个特征，一整个组里都没有一个特征和它配合可以提供更好的信息的话
                    R(1,current_index)=0; %这样就可以把那一个单独的特征选出来啦！！！！
                    MI_Array(1,current_index)=-1;
                else                               
                    I_R=I_current;% 更新最大互信息
                end
            end
%           disp(R);        
end
selectedFeatures=find(R==1); % 最终选中的特征
time=toc(start);
end

