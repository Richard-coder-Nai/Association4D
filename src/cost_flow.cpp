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

void cost_flow::addEdge(const int& fro/*����ߵ����*/, const int& nxt,
	const int& cap, const double& cst) {
	allEdges.push_back(Edge(adjEdgeNo[fro], nxt, cap, cst));
	adjEdgeNo[fro] = allEdges.size() - 1;
	allEdges.push_back(Edge(adjEdgeNo[nxt], fro, 0, -cst));//cst�����ֱ��������
	adjEdgeNo[nxt] = allEdges.size() - 1;
}

bool cost_flow::spfa() {//�ҵ�һ�����·��
	preEdge.resize(v, -1);
	vector<int> flo(v, 2147483647);//����������������v������s,t
	list<int> que;
	que.clear();
	vector<double> dis(v, 1E308);//��costΪȨֵ�����·
	vector<bool> vis(v, 0);
	dis[s] = 0;
	que.push_back(s);
	while( !que.empty()) {
		int current = que.front();//current
		que.pop_front();
		vis[current] = 0;//�����˾���û����
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
						if (!que.empty() && dis[nx] < dis[que.front()]) que.push_front(nx);//���Ը���ıƽ���ȷ��
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
		allEdges[rev = preEdge[current] ^ 1].cap += flo;//��ǰ����ߵı���ǵ�ǰ�ߵı���޸����һλ��0��1������2��3��//��������ߵ�cap����flow,����߼���flow��rev����ߵı��
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
				/*�ҷ����*/
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
	for (; spfa();)//e�ܵ��������
		upd(flow);
}