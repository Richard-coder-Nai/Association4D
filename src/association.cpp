#include "association.h"
#include "math_util.h"
#include "color_util.h"
#include "cost_flow.h"
#include <sstream>
#include <numeric>
#include <algorithm>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include<cstdio>
#include<cstring>
#include<cmath>
#include<list>
#include<vector>



Associater::Associater(const std::vector<Camera>& _cams)
{
	m_cams = _cams;

	m_epiThresh = 0.2f;
	m_wEpi = 1.f;
	m_wView = 4.f;
	m_wPaf = 1.f;
	m_cPlaneTheta = 2e-3f;
	m_cViewCnt = 2.f;
	m_triangulateThresh = 0.05f;
	m_frameMovingThresh = 0.005f;//????
	m_persons2D.resize(m_cams.size());
	m_personsMapByView.resize(m_cams.size());
	m_assignMap.resize(m_cams.size(), std::vector<Eigen::VectorXi>(GetSkelDef().jointSize));
	m_jointRays.resize(m_cams.size(), std::vector<Eigen::Matrix3Xf>(GetSkelDef().jointSize));
	m_jointEpiEdges.resize(GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) 
		m_jointEpiEdges[jIdx].resize(m_cams.size(), std::vector<Eigen::MatrixXf>(m_cams.size()));
}


void Associater::SetDetection(const std::vector<SkelDetection>& detections)
{
	assert(m_cams.size() == detections.size());
	m_detections = detections;

	// reset assign map
	for (int view = 0; view < m_cams.size(); view++)
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
			m_assignMap[view][jIdx].setConstant(m_detections[view].joints[jIdx].cols(), -1);//cols对应人数

	for (int view = 0; view < m_cams.size(); view++) {
		m_persons2D[view].clear();
		m_personsMapByView[view].clear();
	}
	m_persons3D.clear();

}


void Associater::ConstructJointRays()
{
	for (int view = 0; view < m_cams.size(); view++) {
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//加关键点入矩阵
			const Eigen::Matrix3Xf& joints = m_detections[view].joints[jIdx];//列数
			m_jointRays[view][jIdx].resize(3, joints.cols());
			for (int jCandiIdx = 0; jCandiIdx < joints.cols(); jCandiIdx++)
				m_jointRays[view][jIdx].col(jCandiIdx) = m_cams[view].CalcRay(joints.block<2, 1>(0, jCandiIdx));//根据关键点在当前机位的投影构造三维投影线
		}
	}
}


void Associater::ConstructJointEpiEdges()//根据相机参数计算空间距离，背景3，无效值-1，0表最近1最远
{
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
		for (int viewA = 0; viewA < m_cams.size() - 1; viewA++) {
			for (int viewB = viewA + 1; viewB < m_cams.size(); viewB++) {//C(5,2)相机循环
				Eigen::MatrixXf& epi = m_jointEpiEdges[jIdx][viewA][viewB];//同一关键点标号所有无序相机对计算出的约化距离
				const Eigen::Matrix3Xf& jointsA = m_detections[viewA].joints[jIdx];
				const Eigen::Matrix3Xf& jointsB = m_detections[viewB].joints[jIdx];
				const Eigen::Matrix3Xf& raysA = m_jointRays[viewA][jIdx];//方向
				const Eigen::Matrix3Xf& raysB = m_jointRays[viewB][jIdx];
				epi.setConstant(jointsA.cols(), jointsB.cols(), -1.f);
				for (int jaCandiIdx = 0; jaCandiIdx < epi.rows(); jaCandiIdx++) {
					for (int jbCandiIdx = 0; jbCandiIdx < epi.cols(); jbCandiIdx++) {
						const float dist = MathUtil::Line2LineDist(
							m_cams[viewA].pos, raysA.col(jaCandiIdx), m_cams[viewB].pos, raysB.col(jbCandiIdx));
						if (dist < m_epiThresh)
							epi(jaCandiIdx, jbCandiIdx) = dist / m_epiThresh;//[0,1)的标准化越接近1越差
					}
				}
				m_jointEpiEdges[jIdx][viewB][viewA] = epi.transpose();//每个元素都是矩阵，epi是别名
			}
		}
	}
}



void Associater::ClusterPersons2D(const SkelDetection& detection, std::vector<Person2D>& persons, std::vector<Eigen::VectorXi>& assignMap)
{
	persons.clear();

	// generate valid pafs
	std::vector<std::tuple<float, int, int, int>> pafSet;
	for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);//点的类型
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
		for (int jaCandiIdx = 0; jaCandiIdx < detection.joints[jaIdx].cols(); jaCandiIdx++) {//jaCandiIdx,该类型的第jaCandiIndex个点
			for (int jbCandiIdx = 0; jbCandiIdx < detection.joints[jbIdx].cols(); jbCandiIdx++) {//边的端点
				const float jaScore = detection.joints[jaIdx](2, jaCandiIdx);//单图连通概率
				const float jbScore = detection.joints[jbIdx](2, jbCandiIdx);
				const float pafScore = detection.pafs[pafIdx](jaCandiIdx, jbCandiIdx);//两点成边的概率，连接优先级，越小概率越大
				if (jaScore > 0.f && jbScore > 0.f && pafScore > 0.f)
					pafSet.emplace_back(std::make_tuple(pafScore, pafIdx, jaCandiIdx, jbCandiIdx));//加入边集，pafSet为一vector
			}
		}
	}
	std::sort(pafSet.rbegin(), pafSet.rend());//边权排序

	// construct bodies use minimal spanning tree
	assignMap.resize(GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < assignMap.size(); jIdx++)
		assignMap[jIdx].setConstant(detection.joints[jIdx].cols(), -1);

	for (const auto& paf : pafSet) {//pafset的迭代器
		const float pafScore = std::get<0>(paf);//tuple的第一个
		const int pafIdx = std::get<1>(paf);
		const int jaCandiIdx = std::get<2>(paf);
		const int jbCandiIdx = std::get<3>(paf);
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);

		int& aAssign = assignMap[jaIdx][jaCandiIdx];
		int& bAssign = assignMap[jbIdx][jbCandiIdx];

		// 1. A & B not assigned yet: Create new person
		if (aAssign == -1 && bAssign == -1) {
			Person2D person;
			person.joints.col(jaIdx) = detection.joints[jaIdx].col(jaCandiIdx);
			person.joints.col(jbIdx) = detection.joints[jbIdx].col(jbCandiIdx);
			person.pafs(pafIdx) = pafScore;
			aAssign = bAssign = persons.size();//给人编号
			persons.emplace_back(person);
		}

		// 2. A assigned but not B: Add B to person with A (if no another B there) 
		// 3. B assigned but not A: Add A to person with B (if no another A there)
		else if ((aAssign >= 0 && bAssign == -1) || (aAssign == -1 && bAssign >= 0)) {
			const int assigned = aAssign >= 0 ? aAssign : bAssign;
			int& unassigned = aAssign >= 0 ? bAssign : aAssign;//引用
			const int unassignedIdx = aAssign >= 0 ? jbIdx : jaIdx;
			const int unassignedCandiIdx = aAssign >= 0 ? jbCandiIdx : jaCandiIdx;

			Person2D& person = persons[assigned];
			if (person.joints(2, unassignedIdx) < FLT_EPSILON) {//浮点数相等，相减小于epsilon
				person.joints.col(unassignedIdx) = detection.joints[unassignedIdx].col(unassignedCandiIdx);
				person.pafs(pafIdx) = pafScore;
				unassigned = assigned;
			}
		}

		// 4. A & B already assigned to same person (circular/redundant PAF)//环或者冗余
		else if (aAssign == bAssign)
			persons[aAssign].pafs(pafIdx) = pafScore;

		// 5. A & B already assigned to different people: Merge people if key point intersection is null
		else {
			const int assignFst = aAssign < bAssign ? aAssign : bAssign;
			const int assignSec = aAssign < bAssign ? bAssign : aAssign;
			Person2D& personFst = persons[assignFst];
			const Person2D& personSec = persons[assignSec];

			bool conflict = false;
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize && !conflict; jIdx++)//大写Get都是常量
				conflict |= (personFst.joints(2, jIdx) > 0.f && personSec.joints(2, jIdx) > 0.f);

			if (!conflict) {
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
					if (personSec.joints(2, jIdx) > 0.f)
						personFst.joints.col(jIdx) = personSec.joints.col(jIdx);

				persons.erase(persons.begin() + assignSec);
				for (Eigen::VectorXi& tmp : assignMap) {
					for (int i = 0; i < tmp.size(); i++) {
						if (tmp[i] == assignSec)
							tmp[i] = assignFst;
						else if (tmp[i] > assignSec)
							tmp[i]--;
					}
				}
			}
		}
	}

	// filter
	//节点太少的人删掉
	const int jcntThresh = round(0.8f * GetSkelDef().jointSize);
	for (auto person = persons.begin(); person != persons.end();) {
		
		if (person->GetJointCnt() < jcntThresh ) {
			const int personIdx = person - persons.begin();
			for (Eigen::VectorXi& tmp : assignMap) {
				for (int i = 0; i < tmp.size(); i++) {
					if (tmp[i] == personIdx)
						tmp[i] = -1;
					else if (tmp[i] > personIdx)
						tmp[i]--;
				}
			}
			person = persons.erase(person);//指向被删除节点的下一个节点的迭代器
		}
		else
			person++;
	}
}


void Associater::ClusterPersons2D()
{
	// cluster 2D
	for (int view = 0; view < m_cams.size(); view++) {
		std::vector<Eigen::VectorXi> assignMap;
		std::vector<Person2D> persons2D;
		ClusterPersons2D(m_detections[view], persons2D, assignMap);
		m_personsMapByView[view].resize(persons2D.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			for (int candiIdx = 0; candiIdx < assignMap[jIdx].size(); candiIdx++) {
				const int pIdx = assignMap[jIdx][candiIdx];
				if (pIdx >= 0)
					m_personsMapByView[view][pIdx][jIdx] = candiIdx;
			}
		}
	}
}


void Associater::ProposalCollocation()//cross views
{
	// proposal persons
	std::function<void(const Eigen::VectorXi&, const int&, std::vector<Eigen::VectorXi>&)> Proposal
		= [&Proposal](const Eigen::VectorXi& candiCnt, const int& k, std::vector<Eigen::VectorXi>& proposals) {
		if (k == candiCnt.size()) {
			return;
		}//candiCnt[k]k号机位有几个2D人
		else if (k == 0) {
			proposals = std::vector<Eigen::VectorXi>(candiCnt[k] + 1, Eigen::VectorXi::Constant(candiCnt.size(), -1));//每一个假设都是机位个数维+1的向量
			for (int i = 0; i < candiCnt[k]; i++)
				proposals[i + 1][k] = i;
			Proposal(candiCnt, k + 1, proposals);//递归调用寻找下一个机位
		}
		else {
			std::vector<Eigen::VectorXi> proposalsBefore = proposals;
			for (int i = 0; i < candiCnt[k]; i++) {
				std::vector<Eigen::VectorXi> _proposals = proposalsBefore;//初值
				for (auto&& _proposal : _proposals)
					_proposal[k] = i;
				proposals.insert(proposals.end(), _proposals.begin(), _proposals.end());
			}
			Proposal(candiCnt, k + 1, proposals);
		}
	};

	m_personProposals.clear();
	Eigen::VectorXi candiCnt(m_cams.size());
	for (int view = 0; view < m_cams.size(); view++)
		candiCnt[view] = int(m_personsMapByView[view].size());
	Proposal(candiCnt, 0, m_personProposals);
}



float Associater::CalcProposalLoss(const int& personProposalIdx)//一个proposal一个loss
{
	const Eigen::VectorXi& proposal = m_personProposals[personProposalIdx];
	bool valid = (proposal.array() >= 0).count() > 1;//-1太多就干掉
	if (!valid)
		return -1.f;

	float loss = 0.f;

	std::vector<float> epiLosses;//单点loss，joint loss
	for (int viewA = 0; viewA < m_cams.size() - 1 && valid; viewA++) {
		if (proposal[viewA] == -1)
			continue;
		const Eigen::VectorXi& personMapA = m_personsMapByView[viewA][proposal[viewA]];//[view][personIdx][jointsIdx]
		for (int viewB = viewA + 1; viewB < m_cams.size() && valid; viewB++) {
			if (proposal[viewB] == -1)
				continue;
			const Eigen::VectorXi& personMapB = m_personsMapByView[viewB][proposal[viewB]];

			for (int jIdx = 0; jIdx < GetSkelDef().jointSize && valid; jIdx++) {
				if (personMapA[jIdx] == -1 || personMapB[jIdx] == -1)//这个人没拍到这个关键点
					epiLosses.emplace_back(m_epiThresh);
				else {
					const float edge = m_jointEpiEdges[jIdx][viewA][viewB](personMapA[jIdx], personMapB[jIdx]);
					if (edge < 0.f)
						valid = false;
					else
						epiLosses.emplace_back(edge);
				}
			}
		}
	}
	if (!valid)
		return -1.f;

	if (epiLosses.size() > 0)
		loss += m_wEpi * std::accumulate(epiLosses.begin(), epiLosses.end(), 0.f) / float(epiLosses.size());//求平均epiloss里的元素全部做一个平均然后乘一个权值

	// paf loss
	std::vector<float> pafLosses;//边loss
	for (int view = 0; view < m_cams.size() && valid; view++) {
		if (proposal[view] == -1)
			continue;
		const Eigen::VectorXi& personMap = m_personsMapByView[view][proposal[view]];
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
			const Eigen::Vector2i candi(personMap[GetSkelDef().pafDict(0, pafIdx)], personMap[GetSkelDef().pafDict(1, pafIdx)]);
			if (candi.x() >= 0 && candi.y() >= 0)
				pafLosses.emplace_back(1.f - m_detections[view].pafs[pafIdx](candi.x(), candi.y()));
			else
				pafLosses.emplace_back(1.f);
		}
	}
	if (pafLosses.size() > 0)
		loss += m_wPaf * std::accumulate(pafLosses.begin(), pafLosses.end(), 0.f) / float(pafLosses.size());

	// view loss
	loss += m_wView * (1.f - MathUtil::Welsch(m_cViewCnt, (proposal.array() >= 0).count()));
	return loss;
};


void Associater::ClusterPersons3D()
{
	//m_personsMapByIdx.clear();

	// cluster 3D
	std::vector<std::pair<float, int>> losses;
	for (int personProposalIdx = 0; personProposalIdx < m_personProposals.size(); personProposalIdx++) {
		const float loss = CalcProposalLoss(personProposalIdx);
		if (loss > 0.f)
			losses.emplace_back(std::make_pair(loss, personProposalIdx));//第几号proposal的loss是多少
	}

	// parse to cluster greedily
	std::sort(losses.begin(), losses.end());
	std::vector<Eigen::VectorXi> availableMap(m_cams.size());
	for (int view = 0; view < m_cams.size(); view++)
		availableMap[view] = Eigen::VectorXi::Constant(m_personsMapByView[view].size(), 1);//第view个相机的第i个人是否被分配过

	for (const auto& loss : losses) {
		const Eigen::VectorXi& personProposal = m_personProposals[loss.second];//make_pair(loss, personProposalIdx)

		bool available = true;
		for (int i = 0; i < personProposal.size() && available; i++)
			available &= (personProposal[i] == -1 || availableMap[i][personProposal[i]]);

		if (!available)
			continue;

		std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int view = 0; view < m_cams.size(); view++)
			if (personProposal[view] != -1) {//personmap拍到了这个人
				personMap[view] = m_personsMapByView[view][personProposal[view]];
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
					const int candiIdx = personMap[view][jIdx];
					if (candiIdx >= 0)
						m_assignMap[view][jIdx][candiIdx] = m_personsMapByIdx.size();
				}
				availableMap[view][personProposal[view]] = false;
			}
		if(CalcPersonScore(personMap)>8)
			m_personsMapByIdx.emplace_back(personMap);
	}

	// add remain persons
	for (int view = 0; view < m_cams.size(); view++) {
		for (int i = 0; i < m_personsMapByView[view].size(); i++) {
			if (availableMap[view][i]) {
				std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
				personMap[view] = m_personsMapByView[view][i];
				m_personsMapByIdx.emplace_back(personMap);
			}
		}
	}
}


void Associater::ConstructPersons()
{
	// 2D
	for (int view = 0; view < m_cams.size(); view++) {
		m_persons2D[view].clear();
		for (int pIdx = 0; pIdx < m_personsMapByIdx.size(); pIdx++) {
			const std::vector<Eigen::VectorXi>& personMap = *std::next(m_personsMapByIdx.begin(),pIdx);//[pidx]
			const SkelDetection& detection = m_detections[view];
			Person2D person;
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
				if (personMap[view][jIdx] != -1)
					person.joints.col(jIdx) = detection.joints[jIdx].col(personMap[view][jIdx]);

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (personMap[view][jaIdx] != -1 && personMap[view][jbIdx] != -1)
					person.pafs[pafIdx] = detection.pafs[pafIdx](personMap[view][jaIdx], personMap[view][jbIdx]);//只是一个置信度
			}
			m_persons2D[view].emplace_back(person);
		}
	}

	// 3D
	m_persons3D.clear();
	for (int personIdx = 0; personIdx < m_personsMapByIdx.size(); personIdx++) {
		Person3D person;//
		const std::vector<Eigen::VectorXi>& personMap = *std::next(m_personsMapByIdx.begin(), personIdx);
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			MathUtil::Triangulator triangulator;
			for (int camIdx = 0; camIdx < m_cams.size(); camIdx++) {
				if (personMap[camIdx][jIdx] != -1) {
					triangulator.projs.emplace_back(m_cams[camIdx].proj);//proj投影信息，proj投影中心
					triangulator.points.emplace_back(m_persons2D[camIdx][personIdx].joints.col(jIdx).head(2));//投影得到的
				}
			}
			triangulator.Solve();
			if (triangulator.loss < m_triangulateThresh)
				person.joints.col(jIdx) = triangulator.pos.homogeneous();
			else
				person.joints.col(jIdx).setZero();
		}
		m_persons3D.emplace_back(person);
	}
}


void Associater::NaiveTimeTrackingPerson3D(const std::vector<Person3D> lastPerson3D)
{
	m_personsMapByIdx0.clear();
	std::vector<Eigen::VectorXi> assignMap;
	assignMap.resize(GetSkelDef().jointSize);
	
	for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
	{
		std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));

		for (int view = 0; view < m_cams.size(); view++)
		{
			Person2D person = lastPerson3D[pIdx].ProjSkel(m_cams[view].proj);
			for (int jIdx = 0; jIdx < assignMap.size(); jIdx++)
				assignMap[jIdx].setConstant(m_detections[view].joints[jIdx].cols(), -1);
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
			{
				int finalCandiIdx = -1;
				Eigen::Vector3f lastJoint = person.joints.col(jIdx);
				if (lastJoint(2)>FLT_EPSILON)
				{
					std::vector<std::pair<float, int>> dists;
					for (int jCandiIdx = 0; jCandiIdx < m_detections[view].joints[jIdx].cols(); jCandiIdx++)
					{
						Eigen::Vector3f candiJoint = m_detections[view].joints[jIdx].col(jCandiIdx);
						float dist_tmp = MathUtil::Point2PointDist(candiJoint, lastJoint);
						dists.emplace_back(std::make_pair(dist_tmp, jCandiIdx));
					}
					if (!dists.empty())
					{
						std::sort(dists.begin(), dists.end());
						if (dists[0].first < m_frameMovingThresh)
						{
							finalCandiIdx = dists[0].second;
						}
					}
				}
				personMap[view][jIdx] = finalCandiIdx;
				if (finalCandiIdx != -1&&assignMap[jIdx][finalCandiIdx]>0)
				{
					assignMap[jIdx][finalCandiIdx] = pIdx;
					//m_detections[view].joints[jIdx].col(finalCandiIdx)[2] = -1;
				}

			}
			
		}
		
		
		m_personsMapByIdx0.emplace_back(personMap);
	}
				
				
					
				
		

		
}


void Associater::TimeTrackingPersons6PartitleGraphs(const std::vector<Person3D> lastPerson3D)
{
	m_personsMapByIdx.clear();
	m_personsMapByIdx.resize(lastPerson3D.size());
	for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
	{
		m_personsMapByIdx[pIdx].clear();
	}
	for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
	{
		m_personsMapByIdx[pIdx].resize(m_cams.size(),Eigen::VectorXi::Constant(GetSkelDef().jointSize,-1));
	}

	std::vector<Eigen::VectorXi> assignMap;
	
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		int v = 2, e = 0, s = 0, t = 1;
		for (int view = 0; view < m_cams.size(); view++)
		{
			v += 2 * m_detections[view].joints[jIdx].cols() + 2*lastPerson3D.size();
		}
		v += 2 * lastPerson3D.size();
		t = v - 1;
		cost_flow flow(v, s, t);
		/*初始化点的个数、有向边的个数、源点序号、汇点序号*/
		/*初始化点*/
		/*这一帧检测到的点，需要建立点到candidate的映射，要求一一对应，点值互斥，并且还能映回来*/
		std::map<int, std::pair<int, int>> vertexMap;//<vertexNo,<view,jCandiIdx>>
		std::vector<std::vector<int>> vertexInByLevel;
		std::vector<std::vector<int>> vertexOutByLevel;
		for (int view = 0; view < m_cams.size(); view++)
		{
			std::vector<int> vertexIn;
			std::vector<int> vertexOut;
			for (int jCandiIdx = 0; jCandiIdx < m_detections[view].joints[jIdx].cols(); jCandiIdx++)
			{
				int vertexInIdx = vertexMap.size() + 1;
				vertexMap[vertexInIdx] = std::make_pair(view, jCandiIdx);
				/*t = vertexMap.size() + 1;
				v = vertexMap.size() + 2;*/
				vertexIn.push_back(vertexInIdx);
				int vertexOutIdx = vertexMap.size() + 1;
				vertexMap[vertexOutIdx] = std::make_pair(view, jCandiIdx);
				vertexOut.push_back(vertexOutIdx);
				flow.addEdge(vertexInIdx, vertexOutIdx, 1, 1.);
			}
			for (int miss = 0; miss < lastPerson3D.size(); miss++)
			{
				int vertexInIdx = vertexMap.size() + 1;
				vertexMap[vertexInIdx] = std::make_pair(view, -1);
				/*t = vertexMap.size() + 1;
				v = vertexMap.size() + 2;*/
				vertexIn.push_back(vertexInIdx);
				int vertexOutIdx = vertexMap.size() + 1;
				vertexMap[vertexOutIdx] = std::make_pair(view, -1);
				vertexOut.push_back(vertexOutIdx);
				flow.addEdge(vertexInIdx, vertexOutIdx, 1, 1.);
			}
			vertexInByLevel.push_back(vertexIn);
			vertexOutByLevel.push_back(vertexOut);
		}
		/*上一帧检测到的三维人*/
		std::vector<int> vertex3DIn;
		std::vector<int> vertex3DOut;
		for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
		{
			int vertexInIdx = vertexMap.size() + 1;
			vertexMap[vertexInIdx] = std::make_pair(pIdx, jIdx);
			/*t = vertexMap.size() + 1;
			v = vertexMap.size() + 2;*/
			vertex3DIn.push_back(vertexInIdx);
			int vertexOutIdx = vertexMap.size() + 1;
			vertexMap[vertexOutIdx] = std::make_pair(pIdx, jIdx);
			/*t = vertexMap.size() + 1;
			v = vertexMap.size() + 2;*/
			vertex3DOut.push_back(vertexOutIdx);
			flow.addEdge(vertexInIdx, vertexOutIdx, 1, 1.);
		}
		t = vertexMap.size() + 1;
		v = vertexMap.size() + 2;
		vertexInByLevel.push_back(vertex3DIn);
		vertexOutByLevel.push_back(vertex3DOut);
		

		/*初始化边，view与view+1之间，cost为相邻帧的对极距离*/
		for (int level = 0; level < vertexOutByLevel.size()-1; level++)
		{
			if (level < vertexOutByLevel.size() - 2)
			{
				for (int fro = 0; fro<vertexOutByLevel[level].size(); fro++)
					for (int nxt = 0; nxt < vertexInByLevel[level + 1].size(); nxt++)
					{

						int jaCandiIdx = vertexMap[vertexOutByLevel[level][fro]].second;
						int viewA = vertexMap[vertexOutByLevel[level][fro]].first;
						int jbCandiIdx = vertexMap[vertexInByLevel[level + 1][nxt]].second;
						int viewB= vertexMap[vertexInByLevel[level + 1][nxt]].first;
						double cost = -1.;
						if (jaCandiIdx == -1 || jbCandiIdx == -1)
						{
							
								cost = 0.8;
						
							
						}
						else
						{
							cost = m_jointEpiEdges[jIdx][viewA][viewB](jaCandiIdx, jbCandiIdx);
						}
						if (cost > 0.f)
						{
							flow.addEdge( vertexOutByLevel[level][fro], vertexInByLevel[level + 1][nxt], 1, cost);
							
						}

					}
			}
			else
			{
				for (int fro = 0; fro<vertexOutByLevel[level].size(); fro++)
					for (int nxt = 0; nxt < vertexInByLevel[level + 1].size(); nxt++)
					{
						int view = vertexMap[vertexOutByLevel[level][fro]].first;
						int jCandiIdx = vertexMap[vertexOutByLevel[level][fro]].second;
						Eigen::Vector4f pJoint3D = lastPerson3D[vertexMap[vertexInByLevel[level + 1][nxt]].first].joints.col(jIdx);
						double cost = -1;
						if (pJoint3D[3] != 0)
							flow.addEdge(vertexOutByLevel[level][fro], vertexInByLevel[level + 1][nxt], 1, 1.);
						//if (jCandiIdx == -1)
						//{
						//	cost = 1.;
						//	/*Try employ reproj as cost*/
						//	//cost = 0.005;
						//}
						//else if (pJoint3D[3] != 0)
						//{
						//	cost = MathUtil::Point2LineDist(pJoint3D.topRows(3), m_cams[view].pos, m_jointRays[view][jIdx].col(jCandiIdx));
						//	if(cost<m_epiThresh)
						//		cost /= m_epiThresh;
						//	else cost = -1.;
						//	/*Eigen::Vector3f lastJoint = lastPerson3D[vertexMap[vertexInByLevel[level + 1][nxt]].first].ProjSkel(m_cams[view].proj).joints.col(jIdx);
						//	Eigen::Vector3f candiJoint = m_detections[view].joints[jIdx].col(jCandiIdx);
						//	cost= MathUtil::Point2PointDist(candiJoint, lastJoint);*/

						//}
						//if (cost > 0.f)
						//{
						//	flow.addEdge(vertexOutByLevel[level][fro], vertexInByLevel[level + 1][nxt], 1, cost);
						//	
						//}

					}
			}
		}
		/*连接s和t*/
		/*source和level0的所有点相连*/
		for (int nxt = 0; nxt < vertexInByLevel[0].size(); nxt++)
		{
			flow.addEdge(s, vertexInByLevel[0][nxt], 1, 1.);
			
		}

		/*最后一个level和terminal相连*/
		for (int fro = 0; fro < vertexOutByLevel[m_cams.size()].size(); fro++)
		{
			flow.addEdge(vertexOutByLevel[m_cams.size()][fro], t, 1, 1.);
			
		}
		int view0 = vertexMap[29].first;
		int view1 = vertexMap[30].first;
		int view2 = vertexMap[35].first;
		int view3 = vertexMap[32].first;
		int view4 = vertexMap[31].first;
		int view5 = vertexMap[24].first;
		/*六分图匹配*/
		flow.solveFlow();
		/*得到匹配*/
		std::vector<std::vector<int>> matches = flow.backtracking();
		/*按照pIdx顺序填写personsMapByIdx和assignMap*/
		/*得到pIdx与其在本帧其他视角的匹配*/
		//std::vector<pair<int, vector<int>>> matchesByPerson;
		for (int matchIdx = 0; matchIdx < matches.size(); matchIdx++)
		{
			int vertex3D = matches[matchIdx][0];
			int pIdx = vertexMap[vertex3D].first;
			vector<int> personMatch;
			for (int level = 2; level < matches[matchIdx].size(); level = level + 1)
			{
				int vertexIdx = matches[matchIdx][level];
				int view = vertexMap[vertexIdx].first;
				int jCandiIdx = vertexMap[vertexIdx].second;
				m_personsMapByIdx[pIdx][view][jIdx] = jCandiIdx;
				if(jCandiIdx!=-1)
					m_detections[view].joints[jIdx].col(jCandiIdx)[2] = -1;
			}
		}
		/*std::sort(matchesByPerson.begin(), matchesByPerson.end());
		for (int matchIdx = 0; matchIdx < matchesByPerson.size(); matchIdx++)
		{
			int pIdx = matchesByPerson[matchIdx].first;
		}*/
		/*std::cout << "matches size is" << matches.size() << std::endl;
		for (int k = 0; k < matches.size(); k++)
		{
			std::cout << "joint size is " << matches[k].size() << std::endl;
			std::cout << "last vertex is " << matches[k].back() << std::endl;
		}*/


	}
}

void Associater::JointProposalCollocation(const int& jIdx)
{
	// proposal joints
	std::function<void(const Eigen::VectorXi&, const int&, std::vector<Eigen::VectorXi>&)> Proposal
		= [&Proposal](const Eigen::VectorXi& candiCnt, const int& k, std::vector<Eigen::VectorXi>& proposals) {
		if (k == candiCnt.size()) {
			return;
		}//candiCnt[k]k号机位有几个2D人
		else if (k == 0) {
			proposals = std::vector<Eigen::VectorXi>(candiCnt[k] + 1, Eigen::VectorXi::Constant(candiCnt.size(), -1));//每一个假设都是机位个数维+1的向量
			for (int i = 0; i < candiCnt[k]; i++)
				proposals[i + 1][k] = i;
			Proposal(candiCnt, k + 1, proposals);//递归调用寻找下一个机位
		}
		else {
			std::vector<Eigen::VectorXi> proposalsBefore = proposals;
			for (int i = 0; i < candiCnt[k]; i++) {
				std::vector<Eigen::VectorXi> _proposals = proposalsBefore;//初值
				for (auto&& _proposal : _proposals)
					_proposal[k] = i;
				proposals.insert(proposals.end(), _proposals.begin(), _proposals.end());
			}
			Proposal(candiCnt, k + 1, proposals);
		}
	};

	m_jointProposals.clear();
	Eigen::VectorXi candiCnt(m_cams.size());
	for (int view = 0; view < m_cams.size(); view++)
		candiCnt[view] = int(m_detections[view].joints[jIdx].cols());
	Proposal(candiCnt, 0, m_jointProposals);
}

float Associater::CalcJointProposalLoss(const int& jointProposalIdx,const int& jIdx)
{
	const Eigen::VectorXi& proposal = m_jointProposals[jointProposalIdx];
	bool valid = (proposal.array() >= 0).count() > 1;//-1太多就干掉
	if (!valid)
		return -1.f;

	float loss = 0.f;

	std::vector<float> epiLosses;//单点loss，joint loss
	for (int viewA = 0; viewA < m_cams.size() - 1 && valid; viewA++) 
	{
		for (int viewB = viewA + 1; viewB < m_cams.size() && valid; viewB++) 
		{
			
			
				int a = proposal[viewA], b = proposal[viewB];
				int c = m_jointEpiEdges[jIdx][viewA][viewB].rows(), d = m_jointEpiEdges[jIdx][viewA][viewB].cols();
				if (proposal[viewA] == -1 || proposal[viewB] == -1)//这个人没拍到这个关键点
					epiLosses.emplace_back(m_epiThresh);
				else {
					const float edge = m_jointEpiEdges[jIdx][viewA][viewB](proposal[viewA], proposal[viewB]);
					if (edge < 0.f)
						valid = false;
					else
						epiLosses.emplace_back(edge);
				
			}
		}
	}
	if (!valid)
		return -1.f;

	if (epiLosses.size() > 0)
		loss += m_wEpi * std::accumulate(epiLosses.begin(), epiLosses.end(), 0.f) / float(epiLosses.size());//求平均epiloss里的元素全部做一个平均然后乘一个权值

	
	
	// view loss
	loss += m_wView * (1.f - MathUtil::Welsch(m_cViewCnt, (proposal.array() >= 0).count()));
	return loss;
};

void Associater::MatchJoints3D(const int& jIdx)
{
	m_acceptedProposals.clear();
	//match 3D
	/*对枚举得到的proposals进行打分*/
	std::vector<std::pair<float, int>> losses;
	for (int jointProposalIdx = 0; jointProposalIdx < m_jointProposals.size(); jointProposalIdx++) 
	{
		const float loss = CalcJointProposalLoss(jointProposalIdx,jIdx);
		if (loss > 0.f)
			losses.emplace_back(std::make_pair(loss, jointProposalIdx));
	}

	//parse to match greedily
	/*按loss排列proposals，取loss最小的以后删除与当前proposal矛盾的*/
	std::sort(losses.begin(), losses.end());
	std::vector<Eigen::VectorXi> availableMap(m_cams.size());
	for (int view = 0; view < m_cams.size(); view++)
		availableMap[view] = Eigen::VectorXi::Constant(m_detections[view].joints[jIdx].cols(), 1);

	for (const auto& loss : losses) 
	{
		const Eigen::VectorXi& jointProposal = m_jointProposals[loss.second];//make_pair(loss, personProposalIdx)
		const float proposalLoss = loss.first;
		bool available = true;
		for (int i = 0; i < jointProposal.size() && available; i++)
			
			available &= (jointProposal[i] == -1 || availableMap[i][jointProposal[i]]);//availableMap使用存疑

		if (!available)
		{
			
			continue;
		}

		m_acceptedProposals.emplace_back(std::make_pair(proposalLoss, jointProposal));
		
		for (int view = 0; view < m_cams.size(); view++)
			if(jointProposal[view]!=-1)
		{
			
			availableMap[view][jointProposal[view]] = false;
		}
		
	}
}

void Associater::MatchPersons4D(const std::vector<Person3D> lastPerson3D)
{
	m_personsMapByIdx1.clear();
	m_personsMapByIdx1.resize(lastPerson3D.size());
	for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
	{
		m_personsMapByIdx1[pIdx].clear();
	}
	for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
	{
		m_personsMapByIdx1[pIdx].resize(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
	}
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
	{
		m_jointProposals.clear();
		JointProposalCollocation(jIdx);
		MatchJoints3D(jIdx);
		/*将proposal中的关键点投影到3D空间*/
		/*或将上一帧3D关键点投影到2D空间*/
		std::vector<std::tuple<int,float, Eigen::VectorXf>> Joints3D;//<proposalIdx,triangulatot.loss,3D坐标>
		for (int proposalIdx = 0; proposalIdx < m_acceptedProposals.size(); proposalIdx++)
		{
			MathUtil::Triangulator triangulator;
			Eigen::VectorXi proposal = m_acceptedProposals [proposalIdx].second;
			for (int camIdx = 0; camIdx < m_cams.size(); camIdx++)
			{
				
				if (proposal[camIdx] != -1)
				{
					triangulator.projs.emplace_back(m_cams[camIdx].proj);//proj投影信息，proj投影中心
					triangulator.points.emplace_back(m_detections[camIdx].joints[jIdx].col(proposal[camIdx]).head(2));//投影得到的
					
					
				}
				

			}
			triangulator.Solve();
			/*if (triangulator.loss < m_triangulateThresh)*/
			{
				Joints3D.emplace_back(std::make_tuple(proposalIdx, triangulator.loss,triangulator.pos.homogeneous()));
				
			}
			/*else
				person.joints.col(jIdx).setZero();*/
		}

		std::map<int, map<int, float>> costs2DMapByperson;//[pIdx][propsalIdx]
		for (int proposalIdx = 0; proposalIdx < m_acceptedProposals.size(); proposalIdx++)
			for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
			{
				Person3D lastperson = lastPerson3D[pIdx];
				if (lastperson.joints(3, jIdx) < FLT_EPSILON)
					continue;
				vector < float >dists;
				for (int camIdx = 0; camIdx < m_cams.size(); camIdx++)
				{
					
					
					Eigen::VectorXi proposal = m_acceptedProposals[proposalIdx].second;
					if (proposal[camIdx] != -1)
					{
						Person3D lastperson = lastPerson3D[pIdx];
						if (lastperson.joints(3,jIdx) == 1)
						{
							Eigen::Vector3f lastJoint2D = lastperson.ProjSkel(m_cams[camIdx].proj).joints.col(jIdx);
							Eigen::Vector3f candiJoint = m_detections[camIdx].joints[jIdx].col(proposal[camIdx]);
							float dist = MathUtil::Point2PointDist(candiJoint, lastJoint2D);
							dists.emplace_back(dist);
						}
						
					}
					else dists.emplace_back(m_frameMovingThresh);
				}
				float cost = 0.8*std::accumulate(dists.begin(), dists.end(), 0.f) / float(dists.size())+ 0.2*m_acceptedProposals[proposalIdx].first;
				costs2DMapByperson[pIdx][proposalIdx] = cost;
			}

		std::map<int,map<int,float>> costsMapByperson;//[pIdx][propsalIdx]
		for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
		{
			std::map<int,float> proposalCosts;
			Eigen::Vector4f lastJoint3D = lastPerson3D[pIdx].joints.col(jIdx);
			if (lastJoint3D[3] != 1)
				continue;
			std::vector<pair<float,int>> costs;//<cost,proposalIdx>

			for (int jCandiIdx = 0; jCandiIdx < Joints3D.size(); jCandiIdx++)
			{
				int proposalIdx = get<0>(Joints3D[jCandiIdx]);
				float cost = 0.8*(MathUtil::Point2PointDist3D(get<2>(Joints3D[jCandiIdx]).head(3), lastJoint3D.head(3)))+ 0.2*m_acceptedProposals[proposalIdx].first;
				costs.emplace_back(std::make_pair(cost,proposalIdx));
				proposalCosts[proposalIdx] = cost;
				if (proposalIdx == 0)
				{
					proposalIdx = 0;
				}
			}
			costsMapByperson[pIdx]=proposalCosts;
			/*直接选择cost最小的*/
			/*if (!costs.empty())
			{
				std::sort(costs.begin(), costs.end());
				int proposalIdx = costs[0].second;
				std::cout << "min cost is" << costs[0].first << std::endl;
				std::cout << "max cost is" << costs.back().first << std::endl;
				Eigen::VectorXi finallProposal = m_acceptedProposals[proposalIdx].second;
				for (int view = 0; view < m_cams.size(); view++)
				{
					m_personsMapByIdx[pIdx][view][jIdx] = finallProposal[view];
					if(finallProposal[view]!=-1)
						m_detections[view].joints[jIdx].col(finallProposal[view])[2] = -1;
				}
			}*/
			
			

		}
		/*采用费用流实现时域匹配*/
			/*建图*/
		int v = 2, e = 0, s = 0, t = 1;
		v = 2 + 2 * (Joints3D.size() + lastPerson3D.size()) + 2 * lastPerson3D.size();
		t = v - 1;
		cost_flow flow(v, s, t);
		/*初始化点*/
		std::map<int, int> vertexMap;//<vertexNo,proposalIdx>,<vertexNo,pIdx>

		std::vector<int> vertexIn2D;
		std::vector<int> vertexOut2D;
		for (int jCandIdx = 0; jCandIdx < Joints3D.size(); jCandIdx++)
		{
			int vertexInIdx = vertexMap.size() + 1;
			vertexMap[vertexInIdx] = get<0>(Joints3D[jCandIdx]);

			vertexIn2D.push_back(vertexInIdx);
			int vertexOutIdx = vertexMap.size() + 1;
			vertexMap[vertexOutIdx] = get<0>(Joints3D[jCandIdx]);
			vertexOut2D.push_back(vertexOutIdx);
			flow.addEdge(vertexInIdx, vertexOutIdx, 1, 1.);
		}
		for (int miss = 0; miss < lastPerson3D.size(); miss++)
		{
			int vertexInIdx = vertexMap.size() + 1;
			vertexMap[vertexInIdx] = -1;

			vertexIn2D.push_back(vertexInIdx);
			int vertexOutIdx = vertexMap.size() + 1;
			vertexMap[vertexOutIdx] = -1;
			vertexOut2D.push_back(vertexOutIdx);
			flow.addEdge(vertexInIdx, vertexOutIdx, 1, 1.);
		}
		std::vector<int> vertexIn3D;
		std::vector<int> vertexOut3D;
		for (int pIdx = 0; pIdx < lastPerson3D.size(); pIdx++)
		{
			int vertexInIdx = vertexMap.size() + 1;
			vertexMap[vertexInIdx] = pIdx;

			vertexIn3D.push_back(vertexInIdx);
			int vertexOutIdx = vertexMap.size() + 1;
			vertexMap[vertexOutIdx] = pIdx;

			vertexOut3D.push_back(vertexOutIdx);
			flow.addEdge(vertexInIdx, vertexOutIdx, 1, 1.);
		}
		t = vertexMap.size() + 1;
		v = vertexMap.size() + 2;
		for (int fro = 0; fro<vertexOut2D.size(); fro++)
			for (int nxt = 0; nxt < vertexIn3D.size(); nxt++)
			{

				int proposalIdx = vertexMap[vertexOut2D[fro]];
				int pIdx = vertexMap[vertexIn3D[nxt]];
				double cost = -1;
				if (proposalIdx == -1)
				{
					
					cost = 0.8;
					
				}
				else
				{
					cost = costsMapByperson[pIdx][proposalIdx];
				}
				if (cost > 0.f)
				{
					flow.addEdge(vertexOut2D[fro], vertexIn3D[nxt], 1, cost);

				}

			}
		for (int nxt = 0; nxt < vertexIn2D.size(); nxt++)
		{
			flow.addEdge(s, vertexIn2D[nxt], 1, 1.);

		}
		for (int fro = 0; fro < vertexOut3D.size(); fro++)
		{
			flow.addEdge(vertexOut3D[fro], t, 1, 1.);

		}
		flow.solveFlow();
		std::vector<std::vector<int>> matches = flow.backtracking();
		for (int matchIdx = 0; matchIdx < matches.size(); matchIdx++)
		{
			int vertex3D = matches[matchIdx][0];
			int pIdx = vertexMap[vertex3D];
			
			int vertex2D = matches[matchIdx][3];
			int proposalIdx = vertexMap[vertex2D];
			if (proposalIdx == -1)
				continue;
			Eigen::VectorXi finallProposal = m_acceptedProposals[proposalIdx].second;
			for (int view = 0; view < m_cams.size(); view++)
			{
				m_personsMapByIdx1[pIdx][view][jIdx] = finallProposal[view];
				//if (finallProposal[view] != -1)
					//m_detections[view].joints[jIdx].col(finallProposal[view])[2] = -1;
			}
			
				
			
		}
	}
}
	
float Associater::CalcPersonLoss(const std::vector<Eigen::VectorXi>& personMap)
{
	float loss = 0.;
	std::vector<float> pafLosses;//边loss
	for (int view = 0; view < m_cams.size(); view++) 
	{
		
		
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
			const Eigen::Vector2i candi(personMap[view][GetSkelDef().pafDict(0, pafIdx)], personMap[view][GetSkelDef().pafDict(1, pafIdx)]);
			if (candi.x() >= 0 && candi.y() >= 0)
				pafLosses.emplace_back(1.f - m_detections[view].pafs[pafIdx](candi.x(), candi.y()));
			else
				pafLosses.emplace_back(1.f);
		}
	}
	if (pafLosses.size() > 0)
		loss += m_wPaf * std::accumulate(pafLosses.begin(), pafLosses.end(), 0.f) / float(pafLosses.size());
	return loss;
}

float Associater::CalcPersonScore(const std::vector<Eigen::VectorXi>& personMap)
{
	
	std::vector<float> scores;
	for (int view = 0; view < m_cams.size(); view++)
	{
		
		

			const SkelDetection& detection = m_detections[view];
			Person2D person;
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
				if (personMap[view][jIdx] != -1)
					person.joints.col(jIdx) = detection.joints[jIdx].col(personMap[view][jIdx]);

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (personMap[view][jaIdx] != -1 && personMap[view][jbIdx] != -1)
					person.pafs[pafIdx] = detection.pafs[pafIdx](personMap[view][jaIdx], personMap[view][jbIdx]);//只是一个置信度
			}
			scores.emplace_back(person.CalcScore());
		
	}
	int size = scores.size();
	float score= std::accumulate(scores.begin(), scores.end(), 0.f) / float(scores.size());
	return std::accumulate(scores.begin(), scores.end(), 0.f) / float(scores.size());
}

void Associater::FinishPersonsMap()
{
	m_personsMapByIdx.clear();
	for (int pIdx = 0; pIdx < min( m_personsMapByIdx0.size(),m_personsMapByIdx1.size()); pIdx++)
	{
		std::vector<Eigen::VectorXi> personMap0 = m_personsMapByIdx0[pIdx];
		std::vector<Eigen::VectorXi> personMap1 = m_personsMapByIdx1[pIdx];
		if (CalcPersonScore(personMap0) > CalcPersonScore(personMap1))
		{
			if (CalcPersonScore(personMap0) > 19)
			{
				m_personsMapByIdx.emplace_back(personMap0);
				
			}
		}
		else
		{
			if (CalcPersonScore(personMap1) > 19)
			{
				m_personsMapByIdx.emplace_back(personMap1);
			}
		}
	}
	if (m_personsMapByIdx0.size() < m_personsMapByIdx1.size())
	{
		for (int pIdx = m_personsMapByIdx0.size() + 1; pIdx < m_personsMapByIdx1.size(); pIdx++)
		{
			if (CalcPersonScore(m_personsMapByIdx1[pIdx]) > 19)
				m_personsMapByIdx.emplace_back(m_personsMapByIdx1[pIdx]);
		}
	}
	else if (m_personsMapByIdx0.size() > m_personsMapByIdx1.size())
	{
		for (int pIdx = m_personsMapByIdx1.size() + 1; pIdx < m_personsMapByIdx0.size(); pIdx++)
		{
			if (CalcPersonScore(m_personsMapByIdx0[pIdx]) > 19)
				m_personsMapByIdx.emplace_back(m_personsMapByIdx0[pIdx]);
		}
	}
	for (int pIdx = 0; pIdx < m_personsMapByIdx.size(); pIdx++)
		for(int view=0;view<m_cams.size();view++)
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
			{

				int finalCandiIdx = m_personsMapByIdx[pIdx][view][jIdx];
				if(finalCandiIdx!=-1)
					if (m_detections[view].joints[jIdx].col(finalCandiIdx)[2] == -1)
					{
						for (int pDupIdx = 0; pDupIdx < m_personsMapByIdx.size(); pDupIdx++)
							if (m_personsMapByIdx[pDupIdx][view][jIdx] == finalCandiIdx)
							{
								m_personsMapByIdx[pDupIdx][view][jIdx] = -1;
								break;
							}

						finalCandiIdx = m_personsMapByIdx[pIdx][view][jIdx] = -1;
						
						
					}
				if(finalCandiIdx!=-1)
					m_detections[view].joints[jIdx].col(finalCandiIdx)[2] = -1;
					

				
			}
	
}

