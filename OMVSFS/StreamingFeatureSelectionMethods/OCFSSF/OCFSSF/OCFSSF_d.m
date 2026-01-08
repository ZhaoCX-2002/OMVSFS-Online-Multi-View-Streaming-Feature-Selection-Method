function  [pc,time,CSP]  = OCFSSF_d(data1,target_feature,alpha,test)
features = data1(:, 1:target_feature-1);  % 前 p-1 列是特征
labels = data1(:, target_feature);        % 最后一列是标签
min_feat = min(features(:));   % 计算特征最小值
features = features - min_feat + 1;
data1 = [features, labels];
sep=[];
pc=[];
SS=[];
ci=[];%当前的有条件独立节点
cind=[];%有条件独立集合
start=tic;
[n,numFeatures] = size(data1);%得出数据集的实例数，特征数
CSP=cell(1,numFeatures);%定义存放配偶节点集合的元胞数组
ccond=cell(1,numFeatures);%定义存放有条件独立节点对应条件的集合的元胞数组
CPC=cell(1,numFeatures);
ns=max(data1);%节点的大小
selected_features=[];
selected_features1=[];
%target_data=data(:,target_feature);%提取出含有目标属性的那一列数据
%data(:,target_feature)=[]:%删掉原数据的目标信息的一列
%data=[data;target_data];%目标数据的那一列放到数据的最后一行
for i = 1:numFeatures 
    if i==target_feature%是目标特征不考虑
        continue;
    end
    %for very sparse data
    n1=sum(data1(:,i));
    if n1==0
        continue;
    end 
    stop=0;
    [CI,SS]=my_cond_indep_chisquare(data1,i,target_feature,[],test,alpha, ns);
    if CI==1 %与目标属性无条件独立的节点可能是配偶节点
            cind=[cind,i];
            ci=i;
            ccond{i}=[];
   else%无条件相关的节点可能是PC节点，直接进行判断
       stop=1;
   end
   if stop  %首先判断是不是PC节点，不是PC节点再判断是不是配偶节点 
        if ~isempty(selected_features)
            [CI,mcond]=compter_dep_2(selected_features,i,target_feature,3,1, alpha, test,data1);
        end              
        if CI==0  %在CPC（T）条件下，新进入的节点与T相关的，  
            selected_features=[selected_features,i];      
            p2=length(selected_features);
            selected_features1=selected_features;
            if ~isempty(p2)
                 for j=1:p2
                     P=eq(i,selected_features(j));
                     if P==0%判断的节点不是新进入的节点
                        b=setdiff(selected_features1,selected_features(j), 'stable');
                        if ~isempty(b)
                            [CI3,SS1]=optimal_compter_dep_2(b,selected_features(j),target_feature,3, 1, alpha, test,data1);%判断有没有因为新节点的进入，其他节点变得无关，如果无关则不考虑是PC节点，考虑该节点是否为配偶节点
                             if CI3==1   %之前的节点变得独立，从CPC集合中移除，考虑该节点是不是配偶节点
                             %current_feature=setdiff(current_feature,max_feature, 'stable');
                                 selected_features1=b;%剩下的节点集合为PC集合
%                                  fprintf("不是PC的节点可能是配偶节点");
%                                  fprintf("%d\n",selected_features(j));                               
                                 CSP{selected_features(j)}=[];%去掉不可能是PC集合的节点，其找见的相应的配偶节点也不会是配偶节点
    %                              L = ismember(selected_features(j),Unrelated_featurefeature);%没有该无关的节点 %与T条件独立的节点可能是配偶节点
    %                              if L==0%没有该无关的节点
    %                                 Unrelated_featurefeature=[Unrelated_featurefeature,selected_features(j)];
                                    p2=length(selected_features1);%当前的CPC集合
                                    for k=1:p2
                                        KK=ismember(selected_features1(k),CPC{selected_features(j)});
                                        KK=double(KK);
                                        if KK==0
                                            SS=[SS1,selected_features1(k)];%上一步已经找见了与目标特征无关的条件集，加入PC节点，判断是否相关
                                            SS=unique(SS);
                                            CI=my_cond_indep_chisquare(data1,selected_features(j),target_feature,SS,test,alpha, ns);  %目标节点与该节点条件相关 
                                            if CI==0   %该节点在sep并一PC节点的条件下变得相关 
%                                                  N = ismember(selected_features(j),CSP{selected_features1(k)});
%                                                  if N==0%不是其中的节点  
                                                     CSP{selected_features1(k)}=[CSP{selected_features1(k)},selected_features(j)];%新加入配偶节点
                                                     csp12=CSP{selected_features1(k)};%PC节点selected_features1(k)的配偶集合，因为有了新节点的进入，判断之前节点有没有因为新节点的进入与目标特征变得无关，如果无关去掉该节点。
                                                     nn=length(csp12);%配偶集合长度
                                                     csp=csp12;
                                                     if ~isempty(nn)
                                                        for a=1:nn 
                                                            %CPC并与当前PC节点相关的配偶集合
                                                            m1=[selected_features1,csp];%条件集合为所有PC集合并当前PC节点的所有配偶节点
                                                            m1=setdiff(m1,csp12(a),'stable');
                                                            [CI,S,dep]=optimal_compter_dep_21(m1,csp12(a),target_feature,3, 1, alpha, test,data1,selected_features1(k));%判断每个节点与目标特征是否无关，如果无关，则去掉
                                                            if CI==1 || isnan(dep) 
                                                                    CSP{selected_features1(k)}(CSP{selected_features1(k)}==csp12(a))=[];                               
                                                                    csp=CSP{selected_features1(k)};
                                                            end
                                                        end
                                                     end                                            
%                                                  end
                                            end
                                        end
                                    end
                                    CPC{selected_features(j)}=[CPC{selected_features(j)},selected_features1];
                            end
                        end                       
                     end 
               end
            end

        else %在CPC（T）条件下，新进入的节点与T无关的，无条件相关，有条件无关
                cind=[cind,i];
                ccond{i}=mcond;
                ci=i;
        end
   end
   pc=selected_features1;  
%    if ~isempty(pc) && ~isempty(cind) &&(~isempty(dpc)||~isempty(cd))%有条件独立的节点判断
    if ~isempty(pc) && ~isempty(cind)      
      GG=max(pc);%当前pc集合中的最大值
      if ci<GG
         plength1=length(cind);
         for k1=1:plength1
             scind=[ccond{cind(k1)},GG];
             scind=unique(scind);
             CI5=my_cond_indep_chisquare(data1,cind(k1),target_feature,scind,test,alpha, ns);
             if CI5==0%加入PC节点相关 可能为配偶节点
                 CSP{GG}=[CSP{GG},cind(k1)];
                 csp1=CSP{GG};
                 nn=length(csp1);
                 csp=csp1;
                 for b=1:nn 
                    m2=[pc,csp];%CPC并与当前PC节点相关的配偶集合
                    m2=setdiff(m2,csp1(b),'stable');
                    [CI,s,dep]=optimal_compter_dep_21(m2,csp1(b),target_feature,3,1, alpha, test,data1,GG); %判断每个节点与目标特征是否独立，如果独立，则去掉
                    if CI==1 || isnan(dep) 
                            CSP{GG}(CSP{GG}==csp1(b))=[];                          
                            csp=CSP{GG};                                 
                    end
                end
             end
         end
      else
         pp=length(pc);
         for e=1:pp
            scind=[ccond{ci},pc(e)];
            scind=unique(scind);
            CI6=my_cond_indep_chisquare(data1,ci,target_feature,scind,test,alpha, ns);
            if CI6==0%加入PC节点相关 可能为配偶节点
                CSP{pc(e)}=[CSP{pc(e)},ci];
                csp1=CSP{pc(e)};
                nn=length(csp1);
                csp=csp1;
                for b=1:nn 
                    m2=[pc,csp];%CPC并与当前PC节点相关的配偶集合
                    m2=setdiff(m2,csp1(b),'stable');
                    [CI,s,dep]=optimal_compter_dep_21(m2,csp1(b),target_feature,3, 1, alpha, test,data1,pc(e)); %判断每个节点与目标特征是否独立，如果独立，则去掉
                    if CI==1 || isnan(dep) 
                            CSP{pc(e)}(CSP{pc(e)}==csp1(b))=[];                          
                            csp=CSP{pc(e)};
                   end
                end
            end
          end
      end
    end
end
time=toc(start);      




