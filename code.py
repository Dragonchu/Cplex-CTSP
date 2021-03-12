import math
import docplex.mp.model as cpx
import matplotlib.pyplot as plt
import numpy as np
#----------初始化参数
num_vertex = 28#总点数（包括仓库、自提点、客户节点）
num_sdf = 4#自提点个数
num_salesman = 3#快递员个数
num_group = 4#簇的个数
position_vertex_list = np.array([[970.0,1340.0],[40,858],[607,1246],[907,1978],[1448,1580],[1150.0,1760.0],
                                 [630.0,1660.0],[40.0,2090.0],[750.0,1100.0],[750.0,2030.0],[1030.0,2070.0],
                                 [1650.0,650.0],[1490.0,1630.0],[790.0,2260.0],[710.0,1310.0],[840.0,550.0],[1170.0,2300.0],
                                 [370.0,1200.0],[510.0,700.0],[750.0,900.0],[1280.0,1200.0],[230.0,590.0],[460.0,860.0],
                                 [830.0,1770.0],[1840.0,1240.0],[1460.0,1420.0],[240.0,842.0],[-267.0,947.0]])#依次为仓库、自提点、客户节点
position_depot = np.array([970.0,1340.0])
position_sdf_list = np.array([[40,858],[607,1246],[907,1978],[1448,1580]])
position_customers_list = np.array([[1150.0,1760.0],
                                    [630.0,1660.0],[40.0,2090.0],[750.0,1100.0],[750.0,2030.0],[1030.0,2070.0],
                                    [1650.0,650.0],[1490.0,1630.0],[790.0,2260.0],[710.0,1310.0],[840.0,550.0],[1170.0,2300.0],
                                    [370.0,1200.0],[510.0,700.0],[750.0,900.0],[1280.0,1200.0],[230.0,590.0],[460.0,860.0],
                                    [830.0,1770.0],[1840.0,1240.0],[1460.0,1420.0],[240,842],[-267,947]])
radius_capacity_sdf_list = np.array([[400,2],[100,9],[210,6],[200,4]])
#序号以position_vertex_list中响应坐标的下标为准
responsible_salesman_list = np.array([[1,3,18,21,5,7,9,10,13,23,26,27],[2,6,8,11,14,15,17,19,20,22],[4,12,16,24,25]])
responsible_salesman_list_and_depot = np.array([[0,1,3,18,21,5,7,9,10,13,23,26,27],[0,2,6,8,11,14,15,17,19,20,22],[0,4,12,16,24,25]])
group_vertex_index_in_position_vertex_list =np.array([[1,18,21,26,27],[2,6,8,11,14,15,17,19,20,22],[3,5,7,9,10,13,23],[4,12,16,24,25]])
length_of_each_Vk_list = np.array([5,10,7,5])
k_means_labels = np.array([2,1,2,1,2,2,1,3,2,1,1,3,1,0,1,1,0,1,2,3,3,0,0])
position_vertex_x_list = np.array([position_vertex_list[i][0] for i in range(len(position_vertex_list))])
position_vertex_y_list = np.array([position_vertex_list[i][1] for i in range(len(position_vertex_list))])
#路径权值，默认为距离的平方
c_vars = {(i, j):math.pow(math.pow(position_vertex_list[i][0] - position_vertex_list[j][0], 2) +
                          math.pow(position_vertex_list[i][1] - position_vertex_list[j][1], 2), 0.5)
          for i in range(num_vertex) for j in range(num_vertex)}
#各种集合
set_without_depot = range(1,num_vertex)
set_Vs_c = range(1, num_vertex)
set_V = range(num_vertex)
set_Vs = range(1,num_sdf+1)
set_Vc = range(num_sdf+1,num_vertex)
set_salesman = range(0,num_salesman)
set_group = range(0,num_group)
set_sdf_r_c = range(0, num_sdf)
#model
opt_model = cpx.Model(name="CSP Model")
#----------决策变量
x_vars  = {(i,j): opt_model.integer_var(lb=0, ub= 1,name="x_{0}_{1}".format(i,j)) for i in set_V for j in set_V}
z_vars  = {(i,j): opt_model.integer_var(lb=0, ub= 1,name="z_{0}_{1}".format(i,j)) for i in set_Vc for j in set_Vs}
u_vars = {(i,j):opt_model.integer_var(lb=0, ub=len(responsible_salesman_list[j]) - 1, name="u_{0}_{1}".format(i, j))
          for i in set_V for j in set_salesman if i in responsible_salesman_list[j]}
y_vars = {i: opt_model.integer_var(lb=0, ub= 1,name="y_{0}".format(i)) for i in set_V}
#----------约束条件
#约束1 保证n个配送员从仓库出发，最后返回仓库
constraints_1_1 = {j:opt_model.add_constraint(ct=opt_model.sum(x_vars[(0,j)] for j in set_Vs_c) == num_salesman,ctname='约束1_1_{0}'.format(j)) for j in set_Vs_c}
constraints_1_2 = {i:opt_model.add_constraint(ct=opt_model.sum(x_vars[(i,0)] for i in set_Vs_c) == num_salesman,ctname='约束1_2_{0}'.format(i)) for i in set_Vs_c}

#约束2 保证每个客户被遍历一次或者被一个自提点覆盖，保证每个自提点最多被遍历一次
for i in set_Vc:
    opt_model.add_constraint(ct=opt_model.sum(x_vars[(i,j)] for j in set_V if j != i)+opt_model.sum(
        z_vars[(i,l)] for l in set_Vs if l != i)==1,ctname='约束2_1{0}'.format(i))
for i in set_Vs:
    opt_model.add_constraint(ct=opt_model.sum(x_vars[(i, j)] for j in set_V if j != i) <= 1, ctname='约束2_2{0}'.format(i))

# #约束3
# for k in set_group:
#     opt_model.add_constraint(ct=opt_model.sum(x_vars[(i,j)]
#                                               for i in group_vertex_index_in_position_vertex_list[k]
#                                               for j in group_vertex_index_in_position_vertex_list[k] if i !=j)
#                                 <= length_of_each_Vk_list[k]-1
#                              , ctname='约束3_{0}'.format(k))
# #
#约束4 保证覆盖客户的自提点被遍历到
for i in set_Vc:
    for l in set_Vs:
        if i!=l:
            opt_model.add_constraint(ct=opt_model.sum(x_vars[(m,l)] for m in set_V if m != l)
                                        +opt_model.sum(x_vars[(l,m)] for m in set_V if m!=l)>=z_vars[(i,l)]
                                     ,ctname='约束4_{0}_{1}'.format(i,l))

#约束5 保证每一个自提点满足入度和出度为1；保证每个客户在路径上时入度和出度为1，被覆盖时不被遍历
for k in set_salesman:
    for i in responsible_salesman_list[k]:
        if i in set_Vc:
            opt_model.add_constraint(
                ct=opt_model.sum(x_vars[(i,j)] for j in responsible_salesman_list_and_depot[k] if j!=i)
                   ==(1-opt_model.sum(z_vars[(i,l)] for l in set_Vs if l !=i))*1
                ,ctname='约束5_1{0}'.format(i))
            opt_model.add_constraint(
                ct=opt_model.sum(x_vars[(m,i)] for m in responsible_salesman_list_and_depot[k] if m != i)
                   == (1 - opt_model.sum(z_vars[(i, l)] for l in set_Vs if l != i)) * 1
                , ctname='约束5_2{0}'.format(i))
        elif i in set_Vs:
            opt_model.add_constraint(
                ct=opt_model.sum(x_vars[(i, j)] for j in responsible_salesman_list_and_depot[k] if j != i)==1
                , ctname='约束5_1{0}'.format(i))
            opt_model.add_constraint(
                ct=opt_model.sum(x_vars[(m, i)] for m in responsible_salesman_list_and_depot[k] if m != i)==1
            , ctname = '约束5_2{0}'.format(i))

#约束6 保证对于每一个快递员的行程没有子环
for k in set_salesman:
    for i in set_V:
        for j in set_V:
            if i in responsible_salesman_list[k] and j in responsible_salesman_list[k] and i != j:
                opt_model.add_constraint(ct=u_vars[(i,k)]-u_vars[(j,k)]+(num_vertex-num_salesman-1)*x_vars[(i,j)]
                                            <=num_vertex-num_salesman-2
                                         ,ctname='约束6_{0}'.format(k))
#
#约束7 保证自提点覆盖的客户节点个数不会超过它的容量
for b in set_Vs:
    opt_model.add_constraint(ct=opt_model.sum(z_vars[(i,b)] for i in set_Vc) <= radius_capacity_sdf_list[b-1][1], ctname="约束7_{0}".format(b))

#约束8 保证自提点覆盖的客户都在它的服务范围之内
for b in set_Vs:
    for i in set_Vc:
        if i!=b:
            opt_model.add_constraint(ct=c_vars[(i, b)]*z_vars[(i,b)] <= radius_capacity_sdf_list[b-1][0], ctname="约束8_{0}".format(b))

#目标函数
objective = opt_model.sum(x_vars[i,j] * c_vars[i, j] for i in set_V for j in set_V if i!= j)
opt_model.minimize(objective)
#--------求解模型
opt_model.solve()
opt_model.print_solution()
#--------可视化
plt.figure(figsize=(10,10))
theta = np.linspace(0, 2 * np.pi, 200)
colors = ['#4EACC5', '#FF9C34', '#4E9A06','#eb4034']
for k, col in zip(range(len(position_customers_list)), colors):
    my_members = k_means_labels == k		# my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
    cluster_center = position_sdf_list[k]
    radius = radius_capacity_sdf_list[k][0]
    plt.plot(position_customers_list[my_members, 0], position_customers_list[my_members, 1], 'w',
            markerfacecolor=col, marker='o')	# 将同一类的点表示出来
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', marker='o')	# 将聚类中心单独表示出来
    x = np.cos(theta)*radius+cluster_center[0]
    y = np.sin(theta)*radius+cluster_center[1]
    plt.plot(x, y, color="#34eb52", linewidth=1)
plt.plot(position_depot[0],position_depot[1],'#000000', marker='o')

xs = [0, 0]
ys = [1, 1]
salesman_color = np.array(['red', 'blue', 'green','black'])
for i in set_V:
    for j in set_V:
        if opt_model.solution.get_value(x_vars[(i,j)]) == 1.0:
            xs[0] = position_vertex_x_list[i]
            ys[0] = position_vertex_y_list[i]
            xs[1] = position_vertex_x_list[j]
            ys[1] = position_vertex_y_list[j]
            if np.isin(j, responsible_salesman_list[0]):
                salesman = 0
            elif np.isin(j,responsible_salesman_list[1]):
                salesman = 1
            elif np.isin(j,responsible_salesman_list[2]):
                salesman = 2
            else:
                salesman = 3
            plt.annotate(s='',xy=(position_vertex_x_list[j],position_vertex_y_list[j])
                         , xytext=(position_vertex_x_list[i],position_vertex_y_list[i])
                         , arrowprops=dict(arrowstyle='->',color=salesman_color[salesman]))
        elif i in set_Vc and j in set_Vs and opt_model.solution.get_value(z_vars[(i,j)]) == 1.0:
            xs[0] = position_vertex_x_list[i]
            ys[0] = position_vertex_y_list[i]
            xs[1] = position_vertex_x_list[j]
            ys[1] = position_vertex_y_list[j]
            plt.plot(xs, ys,color='#eb8934',ls='--',linewidth='0.3')

plt.show()