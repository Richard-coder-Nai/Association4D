#pragma once
#include "skel.h"
#include "camera.h"


class Associater
{
public:
	Associater(const std::vector<Camera>& _cams);
	~Associater() = default;
	Associater(const Associater& _) = delete;
	Associater& operator=(const Associater& _) = delete;


	const std::vector<std::vector<Person2D>>& GetPersons2D() const { return m_persons2D; }
	const std::vector<Person3D>& GetPersons3D() const { return m_persons3D; }

	void SetDetection(const std::vector<SkelDetection>& detections);
	void ConstructJointRays();
	void ConstructJointEpiEdges();
	void ClusterPersons2D(const SkelDetection& detection, std::vector<Person2D>& persons, std::vector<Eigen::VectorXi>& assignMap);
	void ClusterPersons2D();
	void ClusterPersons3D();
	void MatchJoints3D(const int& jIdx);//贪心搜索实现crossview连接
	void MatchPersons4D(const std::vector<Person3D> lastPerson3D);//cost flow实现时域连接
	void ProposalCollocation();
	void JointProposalCollocation(const int& jIdx);//在Jointlevel枚举proposals
	float CalcProposalLoss(const int& personProposalIdx);
	float CalcJointProposalLoss(const int& jointProposalIdx, const int& jIdx);//在Jointlevel计算proposals的loss
	float CalcPersonLoss(const std::vector<Eigen::VectorXi>& personMap);//对生成的人进行打分
	float CalcPersonScore(const std::vector<Eigen::VectorXi>& personMap);
	void ConstructPersons();
	void NaiveTimeTrackingPerson3D(const std::vector<Person3D> lastPerson3D);//Naive time tracking方案
	void TimeTrackingPersons6PartitleGraphs(const std::vector<Person3D> lastPerson3D);//费用流六部图匹配（最终未采用）
	void Associater::FinishPersonsMap();//根据打分结果选出建的好的人


private:
	float m_epiThresh;
	float m_wEpi;
	float m_wView;
	float m_wPaf;
	float m_cPlaneTheta;
	float m_cViewCnt;
	float m_triangulateThresh;//0.5f????
	float m_frameMovingThresh;

	std::vector<Camera> m_cams;
	std::vector<SkelDetection> m_detections;
	std::vector<std::vector<Eigen::Matrix3Xf>> m_jointRays;
	std::vector<std::vector<std::vector<Eigen::MatrixXf>>> m_jointEpiEdges;	// m_epiEdge[jIdx][viewA][viewB](jA, jB) = epipolar distance
	std::vector<std::vector<Eigen::VectorXi>> m_personsMapByIdx, m_personsMapByView,m_personsMapByIdx0,m_personsMapByIdx1;//by view: [view][personIdx][jointsIndex]；by index:[person3Dindex][view][jointIdx]，candiIdx
	std::vector<Eigen::VectorXi> m_personProposals;
	std::vector<Eigen::VectorXi> m_jointProposals;//[proposalIdx][view],jCandiIdx
	std::vector<std::pair<float,Eigen::VectorXi>> m_acceptedProposals;//3D连接贪心以后的proposal和loss，[proposalIdx]<loss,proposal>
	std::vector<std::vector<Eigen::VectorXi>> m_assignMap;
	std::vector<std::vector<Person2D>> m_persons2D;
	std::vector<Person3D> m_persons3D;
};

