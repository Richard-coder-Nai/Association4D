#include "cost_flow.h"
#include<cstdio>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<list>
#include<vector>
using namespace std;

void cost_flow::init() {
	adjEdgeNo.resize(v, -1);
	allEdges.clear();
	maxflow = 0;
	totcost = 0.0;
}

void cost_flow::addEdge(const int& fro/*正向边的起点*/, const int& nxt,
	const int& cap, const double& cst) {
	allEdges.push_back(Edge(adjEdgeNo[fro], nxt, cap, cst));
	adjEdgeNo[fro] = allEdges.size() - 1;
	allEdges.push_back(Edge(adjEdgeNo[nxt], fro, 0, -cst));//cst正负分辨正负向边
	adjEdgeNo[nxt] = allEdges.size() - 1;
}

bool cost_flow::spfa() {//找到一条最短路径
	preEdge.resize(v, -1);
	vector<int> flo(v, 2147483647);//流过这个点的流量，v包括了s,t
	list<int> que;
	que.clear();
	vector<double> dis(v, 1E308);//用cost为权值算最短路
	vector<bool> vis(v, 0);
	dis[s] = 0;
	que.push_back(s);
	while( !que.empty()) {
		int current = que.front();//current
		que.pop_front();
		vis[current] = 0;//出队了就是没访问
		for (int i = adjEdgeNo[current]; i!=-1; i = allEdges[i].adj) {
			Edge cuEdge = allEdges[i];
			if (cuEdge.cap) {
				int nx=cuEdge.nxt;
				double tmdis= dis[current] + cuEdge.cst;

				if (dis[nx ] > tmdis) {
					dis[nx] = tmdis;
					preEdge[nx] = i;
					flo[nx] = min(flo[current], cuEdge.cap);
					if (!vis[nx]) {
						vis[nx] = 1;
						if (!que.empty() && dis[nx] < dis[que.front()]) que.push_front(nx);//可以更快的逼近正确答案
						else que.push_back(nx);
					}
				}
			}
		}
	}
	return flow = flo[t], !equalDouble(pathcost = dis[t], 1E308);
}

void cost_flow::upd(const int& flo) {
	int current = t;
	for (int rev; current!=s;) {
		allEdges[preEdge[current]].cap -= flo;
		allEdges[rev = preEdge[current] ^ 1].cap += flo;//当前反向边的编号是当前边的编号修改最后一位（0，1），（2，3）//所有正向边的cap减上flow,反向边加上flow，rev反向边的编号
		current = allEdges[rev].nxt;
	}
	maxflow += flo;
	totcost += flo * pathcost;
}

vector<vector<int>> cost_flow::backtracking()
{
	vector<vector<int>> matches;

	for (int i = adjEdgeNo[t]; i!=-1; i = allEdges[i].adj)
	{

		vector<int>backpath;
		if (allEdges[i].cst<1e-8&&allEdges[i].cap>0)
		{

			int cuVertex = allEdges[i].nxt;
			backpath.push_back(cuVertex);
			while (cuVertex != s)
			{
				int size = allEdges.size();
				/*找反向边*/
				int negEdge = adjEdgeNo[cuVertex];
				for (; negEdge!=-1; negEdge = allEdges[negEdge].adj)
				{
					Edge edgeNow = allEdges[negEdge];
					if (allEdges[negEdge].cst < 1e-8&&allEdges[negEdge].cap>0)
						break;

				}


				cuVertex = allEdges[negEdge].nxt;
				if (cuVertex != s)
					backpath.push_back(cuVertex);


			}
			matches.emplace_back(backpath);

		}

	}
	return matches;
}

void cost_flow::solveFlow()
{
	for (; spfa();)//e总的正向边数
		upd(flow);
}