#pragma once
#include<vector>
using namespace std;


class cost_flow
{
public:
	cost_flow() {};
	cost_flow(int v, int s, int t) :v(v), s(s), t(t) { init(); };
	~cost_flow() {};
	struct Edge {
		int adj;//下一个同起点的边在边表中的序号
		int nxt;//指向的点的序号
		int cap;//容量
		double cst;//费用
		Edge(int adj, int nxt, int cap, double cst) :adj(adj), nxt(nxt), cap(cap), cst(cst) {};
	};
	vector<Edge> allEdges;
	vector<int> adjEdgeNo;//v维，每个节点作为起点的第一条边在边表中的序号
	vector<int> preEdge;//当前点在最短路中的前向边的序号
	int v, e, s, t;
	int flow, maxflow;
	double pathcost, totcost;
	void init();
	void addEdge(const int& fro/*正向边的起点*/, const int& nxt,
		const int& cap, const double& cst);
	bool spfa();
	void upd(const int& flo);
	bool equalDouble(double x, double y) { return x - y < 1e-8 && y - x < 1e-8; }
	void solveFlow();
	vector<vector<int>> backtracking();
};
