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
		int adj;//��һ��ͬ���ı��ڱ߱��е����
		int nxt;//ָ��ĵ�����
		int cap;//����
		double cst;//����
		Edge(int adj, int nxt, int cap, double cst) :adj(adj), nxt(nxt), cap(cap), cst(cst) {};
	};
	vector<Edge> allEdges;
	vector<int> adjEdgeNo;//vά��ÿ���ڵ���Ϊ���ĵ�һ�����ڱ߱��е����
	vector<int> preEdge;//��ǰ�������·�е�ǰ��ߵ����
	int v, e, s, t;
	int flow, maxflow;
	double pathcost, totcost;
	void init();
	void addEdge(const int& fro/*����ߵ����*/, const int& nxt,
		const int& cap, const double& cst);
	bool spfa();
	void upd(const int& flo);
	bool equalDouble(double x, double y) { return x - y < 1e-8 && y - x < 1e-8; }
	void solveFlow();
	vector<vector<int>> backtracking();
};
