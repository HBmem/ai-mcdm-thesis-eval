# Fall 2026 MCDM Evaluation

## Table of Contents

- [1. Overview](#1-overview)
- [2. Participant Preferences](#2-participant-preferences)
  - [2.1 Preference Scales](#21-preference-scales)
    - [2.1.1 Existing 5-Point Preference Scale](#211-existing-5-point-preference-scale)
    - [2.1.2 Existing 7-Point Preference Scale](#212-existing-7-point-preference-scale)
    - [2.1.3 Evaluation Scale: Saaty's Conventional 1–9 AHP Scale](#213-evaluation-scale-saatys-conventional-19-ahp-scale)
  - [2.2 Participant Pairwise Comparison Matrix Construction](#22-participant-pairwise-comparison-matrix-construction)
    - [2.2.1 Example: Constructing a Pairwise Comparison](#221-example-constructing-a-pairwise-comparison)
- [3. Participant Preference Aggregation](#3-participant-preference-aggregation)
  - [3.1 Within-Group Preference Aggregation](#31-within-group-preference-aggregation)
  - [3.2 Across-Group Preference Aggregation](#32-across-group-preference-aggregation)
    - [3.2.1 Stakeholder Voting Power](#321-stakeholder-voting-power)
    - [3.2.2 Weighted Geometric Mean Aggregation](#322-weighted-geometric-mean-aggregation)
  - [3.3 Visual Example of Two-Stage Preference Aggregation](#33-visual-example-of-two-stage-preference-aggregation)
    - [3.3.1 Example Configuration](#331-example-configuration)
    - [3.3.2 Step 1: Participant Preference Matrices](#332-step-1-participant-preference-matrices)
    - [3.3.3 Step 2: Within-Group Aggregation](#333-step-2-within-group-aggregation)
    - [3.3.4 Step 3: Apply Stakeholder Voting Power](#334-step-3-apply-stakeholder-voting-power)
    - [3.3.5 Step 4: Aggregation Flow Diagram](#335-step-4-aggregation-flow-diagram)
- [4. System Evaluation](#4-system-evaluation)
  - [4.1 Custom Policy Scenario](#41-custom-policy-scenario)
    - [4.1.1 Policy Question](#411-policy-question)
    - [4.1.2 Policy Alternatives](#412-policy-alternatives)
    - [4.1.3 Evaluation Criteria](#413-evaluation-criteria)
    - [4.1.4 Stakeholder Groups](#414-stakeholder-groups)
    - [4.1.5 Synthetic Decision Matrix](#415-synthetic-decision-matrix)
  - [4.2 Test Scenario Design](#42-test-scenario-design)
    - [4.2.1 Summary of Test Scenarios](#421-summary-of-test-scenarios)
  - [4.3 Baseline and Preference Boundary Testing](#43-baseline-and-preference-boundary-testing)
    - [4.3.1 Scenario S1: Uniform Stakeholder Preferences](#431-scenario-s1-uniform-stakeholder-preferences)
    - [4.3.2 Scenario S2: Extreme Preference Dominance](#432-scenario-s2-extreme-preference-dominance)
  - [4.4 Stakeholder Heterogeneity and Influence Testing](#44-stakeholder-heterogeneity-and-influence-testing)
    - [4.4.1 Scenario S3: Minority Stakeholder Breakaway](#441-scenario-s3-minority-stakeholder-breakaway)
    - [4.4.2 Scenario S4: Competing Stakeholder Priorities](#442-scenario-s4-competing-stakeholder-priorities)
    - [4.4.3 Scenario S5: Stakeholder Voting Power and Participation Imbalance](#443-scenario-s5-stakeholder-voting-power-and-participation-imbalance)
  - [4.5 Ranking Stability and Sensitivity Testing](#45-ranking-stability-and-sensitivity-testing)
    - [4.5.1 Scenario S6: Removal of the Winning Alternative](#451-scenario-s6-removal-of-the-winning-alternative)
    - [4.5.2 Scenario S7: Removal of an Evaluation Criterion](#452-scenario-s7-removal-of-an-evaluation-criterion)
    - [4.5.3 Scenario S8: Gradual Criterion Weight Perturbation](#453-scenario-s8-gradual-criterion-weight-perturbation)
    - [4.5.4 Scenario S9: Addition and Removal of a Dominated Alternative](#454-scenario-s9-addition-and-removal-of-a-dominated-alternative)
  - [4.6 Judgment Consistency and Reliability Testing](#46-judgment-consistency-and-reliability-testing)
    - [4.6.1 Scenario S10: Inconsistent Stakeholder Judgments](#461-scenario-s10-inconsistent-stakeholder-judgments)
  - [4.7 Evaluation and Comparison Methodology](#47-evaluation-and-comparison-methodology)
    - [4.7.1 Independent Computational Testing Implementation](#471-independent-computational-testing-implementation)
    - [4.7.2 Comparison of Deterministic Results](#472-comparison-of-deterministic-results)
    - [4.7.3 Comparison Across Test Scenarios](#473-comparison-across-test-scenarios)
    - [4.7.4 AI Interpretation Evaluation](#474-ai-interpretation-evaluation)
  - [4.8 Evaluation Objectives](#48-evaluation-objectives)

## 1. Overview

This document sets out the proposed methodology for evaluating PoliInsight, a stakeholder-oriented Multi-Criteria Decision-Making (MCDM) application for policy assessment. The evaluation will examine the correctness of the application's deterministic processing, the stability of its policy alternative rankings under controlled changes, and the accuracy of its AI-generated interpretations. It is a testing plan rather than a report of completed experiments; testing will begin following review and approval of the proposed approach.

The deterministic workflow begins with participant judgments about the relative importance of policy evaluation criteria. Participant pairwise comparison matrices are aggregated within stakeholder groups using the geometric mean. Group matrices are then combined using a voting-power-weighted geometric mean to obtain a session-level comparison matrix. The Analytic Hierarchy Process (AHP) derives criterion weights, which are subsequently used in TOPSIS to rank policy alternatives. Sections 2 and 3 document preference collection, matrix construction, and the two-stage aggregation process.

Section 4 defines a synthetic municipal transportation improvement scenario with five policy alternatives, four evaluation criteria, and five stakeholder groups. Ten controlled test scenarios will vary preference judgments, stakeholder voting powers, participation levels, criteria, and alternatives, while also examining inconsistent judgments. For deterministic validation, PoliInsight's intermediate and final calculations will be compared with those of a separately developed computational test implementation that uses equivalent inputs and mathematical definitions. The same experiments will provide results and context for assessing whether the AI interpretation component accurately describes stakeholder preferences, ranking changes, and limitations.

The evaluation will use Saaty's conventional 1–9 AHP intensity scale for all experimental pairwise judgments. PoliInsight's existing 5-point and 7-point linguistic response scales are documented as part of the application design, but they are not the selected scales for the planned experiments. The sections that follow distinguish existing application behavior, proposed experimental configurations, and expected outcomes that remain to be tested.

## 2. Participant Preferences

Participant preferences are collected through the PoliInsight web application using a series of questions determined by the policy scenario and session configuration. The system supports two preference elicitation methods: **direct criterion rating** and **pairwise comparison**.

**Direct Criterion Rating:** Participants independently evaluate the importance of each criterion using a predefined rating scale. For a policy scenario containing $n$ evaluation criteria, each participant responds to $n$ questions:

$$
Q_{\text{direct}}=n
$$

where $Q_{\text{direct}}$ represents the number of questions required for direct criterion rating.

**Pairwise Comparison:** Participants evaluate the relative importance of two criteria at a time. Each question presents a unique pair of criteria and asks the participant to indicate which criterion is more important and to what degree. The responses are subsequently converted into numerical values and organized into a reciprocal pairwise comparison matrix.

For a policy scenario containing $n$ evaluation criteria, the number of unique pairwise comparisons is:

$$
Q_{\text{pairwise}}=\binom{n}{2}=\frac{n(n-1)}{2}
$$

where $Q_{\text{pairwise}}$ represents the number of questions required for pairwise comparison.

For example, a policy scenario containing four evaluation criteria requires:

$$
Q_{\text{pairwise}}=\frac{4(4-1)}{2}=6
$$

Therefore, each participant must complete six pairwise comparison questions to express their relative preferences across all four criteria.

Although PoliInsight supports both preference elicitation methods, **pairwise comparison is selected for the experimental evaluation** because it aligns with the Analytic Hierarchy Process (AHP), which derives criterion weights from relative importance judgments. Additionally, pairwise comparisons allow the internal consistency of participant judgments to be assessed, providing an additional measure for evaluating preference reliability.

### 2.1 Preference Scales

PoliInsight currently provides two predefined linguistic response scales: a **5-point scale** and a **7-point scale**. In the application, participants select verbal judgments that are mapped to numerical comparisons. A value greater than 1 indicates a preference for the left criterion, a value less than 1 indicates a preference for the right criterion, and a value of 1 represents equal importance. Both existing scales use selected intensities from the AHP fundamental scale and preserve reciprocal comparisons.

The tables below document these **existing application options**. They do not define the response scale selected for the proposed system evaluation.

#### 2.1.1 Existing 5-Point Preference Scale

| Linguistic Preference | Numerical Value | Decimal Approximation |
|---|---:|---:|
| Left criterion is more important | $5$ | 5.0000 |
| Left criterion is slightly more important | $3$ | 3.0000 |
| Both criteria are equally important | $1$ | 1.0000 |
| Right criterion is slightly more important | $1/3$ | 0.3333 |
| Right criterion is more important | $1/5$ | 0.2000 |

#### 2.1.2 Existing 7-Point Preference Scale

| Linguistic Preference | Numerical Value | Decimal Approximation |
|---|---:|---:|
| Left criterion is much more important | $7$ | 7.0000 |
| Left criterion is more important | $5$ | 5.0000 |
| Left criterion is slightly more important | $3$ | 3.0000 |
| Both criteria are equally important | $1$ | 1.0000 |
| Right criterion is slightly more important | $1/3$ | 0.3333 |
| Right criterion is more important | $1/5$ | 0.2000 |
| Right criterion is much more important | $1/7$ | 0.1429 |

#### 2.1.3 Evaluation Scale: Saaty's Conventional 1–9 AHP Scale

For the **planned evaluation**, all pairwise comparison tests will instead use Saaty's conventional **1–9 AHP intensity scale**, including intermediate values and their reciprocals. This choice provides a consistent response range across all ten scenarios, including experiments that require extreme judgments of $9$ and $1/9$.

| Numerical Value | Relative Importance |
|---|---|
| $1$ | Equal importance |
| $3$ | Moderate importance |
| $5$ | Strong importance |
| $7$ | Very strong importance |
| $9$ | Extreme importance |
| $2,4,6,8$ | Intermediate judgments between adjacent levels |
| $1/2,1/3,\ldots,1/9$ | Reciprocal judgments favoring the opposite criterion |

The existing 5-point and 7-point scales are retained in this document to describe PoliInsight's current preference collection design. They are **not** the scales used to generate experimental inputs under the approved evaluation plan. Section 4.2 specifies the 1–9 scale for the test scenarios. The worked example in Section 2.2.1 illustrates reciprocal matrix construction using a comparison value of 7; this value is also permitted on the evaluation scale.

### 2.2 Participant Pairwise Comparison Matrix Construction

Each participant's responses are transformed into a square pairwise comparison matrix representing the relative importance of the evaluation criteria.

For a participant $p$ evaluating $n$ criteria, the pairwise comparison matrix is defined as:

$$
\mathbf{A}^{(p)}=
\begin{bmatrix}
a_{11}^{(p)} & a_{12}^{(p)} & \cdots & a_{1n}^{(p)}\\
a_{21}^{(p)} & a_{22}^{(p)} & \cdots & a_{2n}^{(p)}\\
\vdots & \vdots & \ddots & \vdots\\
a_{n1}^{(p)} & a_{n2}^{(p)} & \cdots & a_{nn}^{(p)}
\end{bmatrix}
$$

where:

- $\mathbf{A}^{(p)}$ represents the pairwise comparison matrix constructed from participant $p$'s responses.
- $n$ represents the number of evaluation criteria.
- $a_{ij}^{(p)}$ represents participant $p$'s assessment of the relative importance of criterion $i$ compared with criterion $j$.

The matrix has dimensions $n \times n$ and satisfies two fundamental properties.

**1. Identity Property**

A criterion compared with itself has equal importance. Therefore, every diagonal element is assigned a value of 1:

$$
a_{ii}^{(p)}=1
\qquad \forall i\in\{1,\ldots,n\}
$$

**2. Reciprocal Property**

For each pair of distinct criteria, the comparison in the opposite direction is represented by the reciprocal of the original value:

$$
a_{ji}^{(p)}=\frac{1}{a_{ij}^{(p)}}
\qquad \forall i\ne j
$$

Consequently, participants only need to provide judgments for the $n(n-1)/2$ unique criterion pairs. The remaining off-diagonal matrix entries are calculated automatically using the reciprocal property.

#### 2.2.1 Example: Constructing a Pairwise Comparison

Consider an illustrative pairwise comparison using a judgment value of 7, which is available on both the existing 7-point application scale and the selected 1–9 evaluation scale.

A participant is presented with a question comparing two evaluation criteria:

- **Left criterion ($L$):** Implementation Cost
- **Right criterion ($R$):** Implementation Time

The participant selects:

**"Left criterion is much more important."**

This response corresponds to a numerical value of 7 on both scales.

The system assigns this value to the matrix entry representing the importance of Implementation Cost relative to Implementation Time:

$$
a_{LR}^{(p)}=7
$$

Using the reciprocal property, the reverse comparison is calculated automatically:

$$
a_{RL}^{(p)}=\frac{1}{a_{LR}^{(p)}}=\frac{1}{7}
$$

The resulting comparison entries can be represented as:

$$
\begin{array}{c|cc}
 & \text{Cost} & \text{Time}\\
\hline
\text{Cost} & 1 & 7\\
\text{Time} & \frac{1}{7} & 1
\end{array}
$$

This indicates that the participant considers Implementation Cost to be seven times as important as Implementation Time on the adopted AHP judgment scale.

The same procedure is repeated for every unique pair of evaluation criteria until the participant's complete $n\times n$ reciprocal comparison matrix has been constructed.

For a policy scenario containing four evaluation criteria, the resulting matrix takes the form:

$$
\mathbf{A}^{(p)}=
\begin{bmatrix}
1 & a_{12}^{(p)} & a_{13}^{(p)} & a_{14}^{(p)}\\
\frac{1}{a_{12}^{(p)}} & 1 & a_{23}^{(p)} & a_{24}^{(p)}\\
\frac{1}{a_{13}^{(p)}} & \frac{1}{a_{23}^{(p)}} & 1 & a_{34}^{(p)}\\
\frac{1}{a_{14}^{(p)}} & \frac{1}{a_{24}^{(p)}} & \frac{1}{a_{34}^{(p)}} & 1
\end{bmatrix}
$$

The six independent comparison values correspond to the six questions answered by the participant.

Once constructed, these matrices serve as inputs to the subsequent stakeholder preference aggregation and AHP weighting processes.

---

## 3. Participant Preference Aggregation

Following the collection of participant preferences, PoliInsight aggregates the individual pairwise comparison matrices to establish a collective representation of stakeholder priorities. The aggregation process accounts for both individual participant judgments and the relative voting power assigned to each stakeholder group.

The system employs a **two-stage aggregation process**:

1. **Within-group aggregation:** Individual participant matrices are combined to produce a representative pairwise comparison matrix for each stakeholder group.
2. **Across-group aggregation:** The stakeholder group matrices are combined using their assigned voting powers to produce a single aggregate pairwise comparison matrix representing the session.

The resulting aggregate matrix is then processed using the Analytic Hierarchy Process (AHP) to derive the final criterion weights. These weights serve as inputs to the subsequent TOPSIS policy alternative ranking process.

This approach ensures that participant preferences are first represented within their respective stakeholder groups before group-level voting powers are applied.

### 3.1 Within-Group Preference Aggregation

Each participant submits a pairwise comparison matrix representing their judgments regarding the relative importance of the evaluation criteria.

For a participant $p$ belonging to stakeholder group $g$, the individual pairwise comparison matrix is represented as:

$$
\mathbf{A}^{(g,p)} =
\begin{bmatrix}
a_{11}^{(g,p)} & a_{12}^{(g,p)} & \cdots & a_{1n}^{(g,p)} \\
a_{21}^{(g,p)} & a_{22}^{(g,p)} & \cdots & a_{2n}^{(g,p)} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1}^{(g,p)} & a_{n2}^{(g,p)} & \cdots & a_{nn}^{(g,p)}
\end{bmatrix}
$$

Where:

- $g$ identifies the stakeholder group.
- $p$ identifies a participant within group $g$.
- $n$ represents the number of evaluation criteria.
- $a_{ij}^{(g,p)}$ represents participant $p$'s assessment of the relative importance of criterion $i$ compared with criterion $j$.

To combine the responses of participants belonging to the same stakeholder group, PoliInsight applies the **geometric mean aggregation method**.

For stakeholder group $g$ containing $m_g$ participants, each element of the group's aggregate comparison matrix is calculated as:

$$
\boxed{
a_{ij}^{(g)} =
\left(
\prod_{p=1}^{m_g} a_{ij}^{(g,p)}
\right)^{\frac{1}{m_g}}
}
$$

Where:

- $a_{ij}^{(g)}$ is the aggregated comparison value for criteria $i$ and $j$ within group $g$.
- $m_g$ is the number of participants belonging to stakeholder group $g$.
- $a_{ij}^{(g,p)}$ is the comparison value submitted by participant $p$.

The geometric mean combines individual judgments while preserving the reciprocal structure of the AHP comparison matrix.

For example, if two participants provide comparison values of 3 and 1 for the same criterion pair, their group's aggregate comparison is:

$$
a_{ij}^{(g)} = \sqrt{3 \times 1}
$$

$$
a_{ij}^{(g)} \approx 1.7321
$$

The reverse comparison is automatically represented by its reciprocal:

$$
a_{ji}^{(g)} = \frac{1}{1.7321} \approx 0.5774
$$

This process is repeated for every pair of criteria, producing a single aggregated comparison matrix for each stakeholder group.

The resulting group matrix can be represented as:

$$
\mathbf{A}^{(g)} =
\begin{bmatrix}
1 & a_{12}^{(g)} & \cdots & a_{1n}^{(g)} \\
\frac{1}{a_{12}^{(g)}} & 1 & \cdots & a_{2n}^{(g)} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{1}{a_{1n}^{(g)}} & \frac{1}{a_{2n}^{(g)}} & \cdots & 1
\end{bmatrix}
$$

This matrix represents the aggregated preferences of the participants within stakeholder group $g$.

### 3.2 Across-Group Preference Aggregation

After generating the individual stakeholder group matrices, PoliInsight combines them to obtain the collective preferences of the session.

Unlike within-group aggregation, which treats each participant equally within their group, across-group aggregation incorporates the **voting power assigned to each stakeholder group**.

This distinction allows the system to separate stakeholder participation from stakeholder influence.

For example, a group containing twenty participants does not automatically receive greater voting power than a group containing five participants. Instead, each group's contribution to the session aggregate is determined by its configured voting power.

#### 3.2.1 Stakeholder Voting Power

Let $G$ represent the number of active stakeholder groups participating in a session.

Each stakeholder group $g$ is assigned a nonnegative voting power $v_g$, normalized such that:

$$
\boxed{
\sum_{g=1}^{G} v_g = 1
}
$$

Where:

- $v_g$ represents the normalized voting power of stakeholder group $g$.
- $G$ represents the number of active stakeholder groups.

If the configured voting powers do not sum to 1, they are normalized before aggregation:

$$
v_g = \frac{\tilde{v}_g}{\sum_{h=1}^{G}\tilde{v}_h}
$$

Where $\tilde{v}_g$ represents the original voting power assigned to group $g$.

Only groups included in the active session aggregation are considered during normalization. The total configured voting power of these groups must be greater than zero.

#### 3.2.2 Weighted Geometric Mean Aggregation

The stakeholder group matrices are combined using the weighted geometric mean.

For each pair of criteria $i$ and $j$, the session-level aggregate comparison is calculated as:

$$
\boxed{
a_{ij}^{(S)} =
\prod_{g=1}^{G}
\left(a_{ij}^{(g)}\right)^{v_g}
}
$$

Where:

- $a_{ij}^{(S)}$ represents the session-level aggregate comparison value.
- $a_{ij}^{(g)}$ represents the aggregated comparison value for stakeholder group $g$.
- $v_g$ represents the normalized voting power of stakeholder group $g$.
- $G$ represents the number of active stakeholder groups.

The resulting session-level matrix is represented as:

$$
\mathbf{A}^{(S)} =
\begin{bmatrix}
1 & a_{12}^{(S)} & \cdots & a_{1n}^{(S)} \\
\frac{1}{a_{12}^{(S)}} & 1 & \cdots & a_{2n}^{(S)} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{1}{a_{1n}^{(S)}} & \frac{1}{a_{2n}^{(S)}} & \cdots & 1
\end{bmatrix}
$$

By applying voting power at the stakeholder group level rather than directly to individual participants, the aggregation procedure ensures that differences in group participation do not automatically translate into differences in formally assigned voting power.

---

### 3.3 Visual Example of Two-Stage Preference Aggregation

The following example demonstrates how participant preferences are aggregated into stakeholder group matrices and subsequently combined into a session-level matrix.

#### 3.3.1 Example Configuration

Consider a simplified policy evaluation containing three criteria:

| Criterion | Description |
|---|---|
| C1 | Implementation Cost |
| C2 | Expected Benefit |
| C3 | Implementation Time |

The session includes two stakeholder groups, each containing two participants.

| Stakeholder Group | Participants | Voting Power |
|---|---:|---:|
| Group A | 2 | 50% |
| Group B | 2 | 50% |
| **Total** | **4** | **100%** |

Participants provide their preferences using the pairwise comparison method.

For three criteria, each participant answers:

$$
Q_{\text{pairwise}} = \frac{3(3-1)}{2} = 3
$$

Therefore, each participant provides three independent comparison judgments, which are used to construct a $3 \times 3$ reciprocal matrix.

#### 3.3.2 Step 1: Participant Preference Matrices

Suppose the participants submit the following judgments.

**Group A — Participant 1**

This participant considers Implementation Cost more important than the other two criteria, while considering Expected Benefit and Implementation Time equally important.

$$
\mathbf{A}^{(A,1)} =
\begin{bmatrix}
1 & 3 & 3\\
\frac{1}{3} & 1 & 1\\
\frac{1}{3} & 1 & 1
\end{bmatrix}
$$

**Group A — Participant 2**

This participant considers all three criteria equally important.

$$
\mathbf{A}^{(A,2)} =
\begin{bmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1
\end{bmatrix}
$$

**Group B — Participant 1**

This participant considers Implementation Time more important than the other two criteria.

$$
\mathbf{A}^{(B,1)} =
\begin{bmatrix}
1 & 1 & \frac{1}{3}\\
1 & 1 & \frac{1}{3}\\
3 & 3 & 1
\end{bmatrix}
$$

**Group B — Participant 2**

This participant considers all three criteria equally important.

$$
\mathbf{A}^{(B,2)} =
\begin{bmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1
\end{bmatrix}
$$

All four matrices satisfy the reciprocal property. In this constructed example, they are also perfectly consistent.

#### 3.3.3 Step 2: Within-Group Aggregation

Each stakeholder group's participant matrices are combined using the geometric mean.

**Group A Aggregation**

For the comparison between C1 and C2:

$$
a_{12}^{(A)} = \sqrt{3 \times 1} = \sqrt{3}
$$

For the comparison between C1 and C3:

$$
a_{13}^{(A)} = \sqrt{3 \times 1} = \sqrt{3}
$$

For the comparison between C2 and C3:

$$
a_{23}^{(A)} = \sqrt{1 \times 1} = 1
$$

The complete Group A aggregate matrix becomes:

$$
\boxed{
\mathbf{A}^{(A)} =
\begin{bmatrix}
1 & \sqrt{3} & \sqrt{3}\\
\frac{1}{\sqrt{3}} & 1 & 1\\
\frac{1}{\sqrt{3}} & 1 & 1
\end{bmatrix}
}
$$

Its approximate numerical representation is:

$$
\mathbf{A}^{(A)} \approx
\begin{bmatrix}
1 & 1.7321 & 1.7321\\
0.5774 & 1 & 1\\
0.5774 & 1 & 1
\end{bmatrix}
$$

**Group B Aggregation**

The same procedure is applied to Group B.

For the comparison between C1 and C2:

$$
a_{12}^{(B)} = \sqrt{1 \times 1} = 1
$$

For the comparison between C1 and C3:

$$
a_{13}^{(B)} = \sqrt{\frac{1}{3} \times 1} = \frac{1}{\sqrt{3}}
$$

For the comparison between C2 and C3:

$$
a_{23}^{(B)} = \sqrt{\frac{1}{3} \times 1} = \frac{1}{\sqrt{3}}
$$

The complete Group B aggregate matrix becomes:

$$
\boxed{
\mathbf{A}^{(B)} =
\begin{bmatrix}
1 & 1 & \frac{1}{\sqrt{3}}\\
1 & 1 & \frac{1}{\sqrt{3}}\\
\sqrt{3} & \sqrt{3} & 1
\end{bmatrix}
}
$$

Its approximate numerical representation is:

$$
\mathbf{A}^{(B)} \approx
\begin{bmatrix}
1 & 1 & 0.5774\\
1 & 1 & 0.5774\\
1.7321 & 1.7321 & 1
\end{bmatrix}
$$

#### 3.3.4 Step 3: Apply Stakeholder Voting Power

After calculating the stakeholder group matrices, the system combines them using the weighted geometric mean.

Both groups have equal voting power:

$$
v_A = 0.5
$$

$$
v_B = 0.5
$$

The session-level comparison between C1 and C2 is:

$$
a_{12}^{(S)}
=
\left(\sqrt{3}\right)^{0.5}
(1)^{0.5}
$$

$$
a_{12}^{(S)} = 3^{1/4} \approx 1.3161
$$

The comparison between C1 and C3 is:

$$
a_{13}^{(S)}
=
\left(\sqrt{3}\right)^{0.5}
\left(\frac{1}{\sqrt{3}}\right)^{0.5}
$$

$$
a_{13}^{(S)} = 1
$$

The comparison between C2 and C3 is:

$$
a_{23}^{(S)}
=
(1)^{0.5}
\left(\frac{1}{\sqrt{3}}\right)^{0.5}
$$

$$
a_{23}^{(S)} = 3^{-1/4} \approx 0.7598
$$

The resulting session-level aggregate matrix is:

$$
\boxed{
\mathbf{A}^{(S)}
=
\begin{bmatrix}
1 & 3^{1/4} & 1\\
3^{-1/4} & 1 & 3^{-1/4}\\
1 & 3^{1/4} & 1
\end{bmatrix}
}
$$

Its approximate numerical representation is:

$$
\mathbf{A}^{(S)} \approx
\begin{bmatrix}
1 & 1.3161 & 1\\
0.7598 & 1 & 0.7598\\
1 & 1.3161 & 1
\end{bmatrix}
$$

This matrix represents the collective criterion preferences after accounting for both individual participant judgments and stakeholder group voting power.

#### 3.3.5 Step 4: Aggregation Flow Diagram

The complete aggregation process can be visualized as follows:

```text
                  PARTICIPANT PREFERENCES
                  =======================

    STAKEHOLDER GROUP A              STAKEHOLDER GROUP B
    Voting Power: 50%                Voting Power: 50%

    ┌─────────────────┐              ┌─────────────────┐
    │  Participant 1  │              │  Participant 1  │
    │                 │              │                 │
    │  Pairwise       │              │  Pairwise       │
    │  Matrix A(A,1)  │              │  Matrix A(B,1)  │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │  Participant 2  │              │  Participant 2  │
    │                 │              │                 │
    │  Pairwise       │              │  Pairwise       │
    │  Matrix A(A,2)  │              │  Matrix A(B,2)  │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ WITHIN-GROUP    │              │ WITHIN-GROUP    │
    │ AGGREGATION     │              │ AGGREGATION     │
    │                 │              │                 │
    │ Geometric Mean  │              │ Geometric Mean  │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             ▼                                ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ Group A Matrix  │              │ Group B Matrix  │
    │                 │              │                 │
    │    A(A)         │              │    A(B)         │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             │                                │
             └───────────────┬────────────────┘
                             │
                             ▼
               ┌────────────────────────┐
               │ ACROSS-GROUP           │
               │ AGGREGATION            │
               │                        │
               │ Weighted Geometric     │
               │ Mean                   │
               │                        │
               │ Group A Weight = 0.50  │
               │ Group B Weight = 0.50  │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ SESSION AGGREGATE      │
               │ PAIRWISE MATRIX        │
               │                        │
               │         A(S)           │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ AHP WEIGHT CALCULATION │
               │                        │
               │ Criterion Weights      │
               │ Consistency Analysis   │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │ FINAL CRITERION WEIGHTS│
               │                        │
               │ C1: 0.3624             │
               │ C2: 0.2752             │
               │ C3: 0.3624             │
               └────────────┬───────────┘
                            │
                            ▼
                    TOPSIS RANKING
```

*Figure: Two-stage stakeholder preference aggregation and AHP criterion weight calculation.*

The diagram illustrates how the system transforms individual participant judgments into stakeholder group preferences, incorporates group voting power, and produces the final criterion weights used in the ranking process.

Although the example contains only two stakeholder groups, the same procedure applies to sessions containing additional groups and participants.

---

## 4. System Evaluation

The PoliInsight application will be evaluated through a series of controlled experiments designed to examine the mathematical correctness, behavior, and stability of its Multi-Criteria Decision-Making (MCDM) engine under different decision-making conditions.

The evaluation will use a custom policy scenario containing synthetic data and ten test scenarios designed to examine how changes in stakeholder preferences, voting power, participation, evaluation criteria, and policy alternatives influence the resulting criterion weights and alternative rankings.

To validate the deterministic processing, PoliInsight's results will be compared against those generated by an independent computational testing implementation. This implementation will be developed separately from PoliInsight and will reproduce the relevant mathematical procedures without depending on the application's web interface, processing restrictions, or existing calculation functions.

The independent implementation will serve as a reference for examining the correctness of PoliInsight's participant preference matrices, stakeholder aggregation procedures, AHP criterion weights, and TOPSIS rankings. Separating the testing implementation from the application is intended to make the mathematical procedures easier to inspect, test, and verify while reducing the possibility that application-specific restrictions or implementation errors affect both sets of calculations.

The evaluation will also extend to PoliInsight's AI interpretation component. The deterministic results produced during each experiment will provide the basis for assessing whether the AI accurately interprets the policy scenario, stakeholder preferences, ranking changes, and limitations of the evaluation.

This document establishes the proposed evaluation methodology. The experiments and comparisons described in the following sections will be conducted after the evaluation approach has been reviewed and approved.

### 4.1 Custom Policy Scenario

A synthetic municipal transportation planning scenario will be used to evaluate PoliInsight's MCDM engine. The scenario represents a municipal government selecting among competing transportation improvement initiatives while considering multiple, potentially conflicting evaluation criteria.

The use of synthetic data provides a controlled environment in which the performance characteristics of policy alternatives can be modified independently of stakeholder preferences. This allows the experiments to isolate specific variables and examine their effects on criterion weighting, preference aggregation, and alternative ranking.

#### 4.1.1 Policy Question

The policy scenario addresses the following decision-making question:

**How should a municipal government prioritize transportation improvement initiatives when balancing implementation costs, public accessibility, environmental benefits, and implementation time?**

The scenario includes five policy alternatives, four evaluation criteria, and five stakeholder groups representing different perspectives on municipal transportation planning.

#### 4.1.2 Policy Alternatives

Five transportation improvement initiatives are considered as potential policy alternatives.

| ID | Policy Alternative | Description |
|---|---|---|
| A1 | Public Transit Expansion | Expand bus routes and increase service frequency. |
| A2 | Bicycle Infrastructure | Construct protected bicycle lanes and improve bicycle connectivity. |
| A3 | Road Infrastructure Improvements | Repair roads, improve intersections, and upgrade traffic signals. |
| A4 | Pedestrian Accessibility | Improve sidewalks, crosswalks, and pedestrian safety infrastructure. |
| A5 | Integrated Mobility Program | Combine smaller transit, cycling, and pedestrian improvements into one coordinated initiative. |

Each alternative represents a different approach to improving municipal transportation infrastructure. The alternatives vary in implementation requirements and expected benefits, allowing the MCDM engine to evaluate trade-offs among the evaluation criteria.

#### 4.1.3 Evaluation Criteria

The policy alternatives are evaluated using four criteria representing financial, accessibility, environmental, and implementation considerations.

| ID | Criterion | Optimization Direction | Description |
|---|---|---|---|
| C1 | Implementation Cost | Minimize | Total expected project expenditure. |
| C2 | Accessibility Improvement | Maximize | Estimated improvement in transportation accessibility. |
| C3 | Environmental Benefit | Maximize | Estimated environmental improvement resulting from implementation. |
| C4 | Implementation Time | Minimize | Estimated time required to complete the project. |

The criteria include both minimization and maximization objectives.

Implementation Cost and Implementation Time are minimization criteria, meaning that lower values represent more desirable outcomes. Accessibility Improvement and Environmental Benefit are maximization criteria, meaning that higher values represent more desirable outcomes.

This combination allows the evaluation to examine how the MCDM engine processes criteria with different optimization directions.

#### 4.1.4 Stakeholder Groups

Five stakeholder groups are included to represent different perspectives on transportation policy.

| Stakeholder Group | Primary Perspective |
|---|---|
| Community Residents | Accessibility, transportation affordability, and quality of life |
| Transportation Department | Infrastructure effectiveness and implementation feasibility |
| City Government | Budget constraints and public-service outcomes |
| Environmental Advocates | Sustainability and environmental improvement |
| Business Community | Transportation efficiency and economic accessibility |

These groups provide the basis for evaluating stakeholder preference collection, within-group aggregation, and the application of group voting power.

The test scenarios will introduce different preference distributions and participation configurations to examine how stakeholder heterogeneity influences the aggregate criterion weights and policy alternative rankings.

#### 4.1.5 Synthetic Decision Matrix

The policy scenario uses a synthetic decision matrix containing the performance values of each alternative across the four evaluation criteria.

**Table 1. Synthetic Transportation Policy Decision Matrix**

| Alternative | Implementation Cost ($M) ↓ | Accessibility Improvement (0–100) ↑ | Environmental Benefit (0–100) ↑ | Implementation Time (Months) ↓ |
|---|---:|---:|---:|---:|
| A1: Public Transit Expansion | 8.0 | 95 | 75 | 24 |
| A2: Bicycle Infrastructure | 3.0 | 65 | 90 | 12 |
| A3: Road Infrastructure Improvements | 6.0 | 80 | 40 | 18 |
| A4: Pedestrian Accessibility | 2.0 | 70 | 65 | 8 |
| A5: Integrated Mobility Program | 5.0 | 85 | 80 | 16 |

The synthetic data are structured so that each alternative exhibits a different combination of advantages and disadvantages.

Public Transit Expansion (A1) provides the greatest accessibility improvement but requires the highest implementation cost and longest completion time. Bicycle Infrastructure (A2) provides the greatest environmental benefit but offers a comparatively modest accessibility improvement. Road Infrastructure Improvements (A3) provides relatively strong accessibility benefits but has the lowest environmental benefit among the original alternatives.

Pedestrian Accessibility (A4) is the least expensive alternative and has the shortest implementation time. The Integrated Mobility Program (A5) provides comparatively balanced performance across the four criteria without achieving the highest performance on any individual criterion.

These differences create conditions in which the preferred alternative may depend on the relative importance assigned to each evaluation criterion.

For example, a preference configuration emphasizing implementation cost and time may favor Pedestrian Accessibility, while a configuration emphasizing accessibility improvement may favor Public Transit Expansion. Similarly, a configuration prioritizing environmental benefits may favor Bicycle Infrastructure.

These outcomes are not predetermined. The final rankings depend on the criterion weights and the TOPSIS calculations performed during the evaluation.

### 4.2 Test Scenario Design

The system evaluation will consist of ten controlled test scenarios examining different aspects of PoliInsight's deterministic MCDM processing and AI interpretation capabilities.

All experiments involving participant pairwise comparisons will use **Saaty's conventional 1–9 AHP intensity scale**, including its reciprocal values, as defined in Section 2.1.3.

The scale represents the relative importance of two criteria using the following numerical judgments.

| Numerical Value | Interpretation |
|---|---|
| 1 | Equal importance |
| 3 | Moderate importance |
| 5 | Strong importance |
| 7 | Very strong importance |
| 9 | Extreme importance |
| 2, 4, 6, 8 | Intermediate judgments between adjacent intensity levels |
| 1/2 through 1/9 | Reciprocal judgments indicating preference for the opposite criterion |

The use of this scale establishes a consistent method for representing participant preferences across the experiments. It also allows the evaluation to examine a broader range of preference intensities than the previously implemented 5-point and 7-point linguistic scales.

Each test scenario will specify its experimental conditions, the variables being modified, and the expected system behavior. Where applicable, the resulting criterion weights, consistency ratios, TOPSIS closeness coefficients, and alternative rankings will be compared against the baseline configuration and independently calculated reference results.

#### 4.2.1 Summary of Test Scenarios

**Table 2. System Evaluation Test Scenarios**

| ID | Test Scenario | Primary Variable | Evaluation Purpose |
|---|---|---|---|
| S1 | Uniform Stakeholder Preferences | Identical pairwise judgments | Establishes the mathematical baseline. |
| S2 | Extreme Preference Dominance | Maximum and minimum preference values | Evaluates preference boundaries and consistency. |
| S3 | Minority Stakeholder Breakaway | Preferences of one stakeholder group | Measures the influence of stakeholder disagreement. |
| S4 | Competing Stakeholder Priorities | Different stakeholder preference profiles | Evaluates aggregation under heterogeneous preferences. |
| S5 | Voting Power and Participation Imbalance | Stakeholder voting power and group participation | Evaluates the relationship between stakeholder representation and influence. |
| S6 | Removal of Winning Alternative | Available policy alternatives | Evaluates rank reversal and ranking stability. |
| S7 | Removal of Evaluation Criterion | Available evaluation criteria | Measures the influence of individual criteria on rankings. |
| S8 | Criterion Weight Perturbation | Criterion importance weights | Identifies ranking sensitivity and transition thresholds. |
| S9 | Addition and Removal of a Dominated Alternative | Composition of the alternative set | Evaluates TOPSIS robustness. |
| S10 | Inconsistent Stakeholder Judgments | Consistency of pairwise comparisons | Evaluates inconsistency detection and reliability. |

The following subsections describe the configuration and evaluation objectives of each scenario.

### 4.3 Baseline and Preference Boundary Testing

The first two scenarios establish the baseline behavior of the MCDM engine and examine how it handles extreme preference judgments.

#### 4.3.1 Scenario S1: Uniform Stakeholder Preferences

**Objective:** Establish a reference condition in which all stakeholders assign equal importance to every evaluation criterion.

This scenario serves as the baseline for subsequent experiments. It uses five stakeholder groups, each containing one participant. All participants submit identical pairwise comparisons, and every comparison is assigned a value of 1, indicating equal importance between the two criteria.

All stakeholder groups are assigned equal voting power.

For the four evaluation criteria, each participant submits the following pairwise comparison matrix:

$$
\mathbf{A}=
\begin{bmatrix}
1&1&1&1\\
1&1&1&1\\
1&1&1&1\\
1&1&1&1
\end{bmatrix}
$$

Because all criteria are considered equally important, the expected AHP criterion-weight vector is:

$$
\mathbf{W}=[0.25,0.25,0.25,0.25]
$$

The matrix is perfectly consistent, resulting in an expected consistency ratio of:

$$
CR=0
$$

The MCDM engine is expected to generate equal criterion weights, preserve the consistency of the pairwise comparison matrix, and produce a reproducible TOPSIS ranking.

Because all participants submit identical judgments, the individual, stakeholder group, and session-level preference matrices should agree.

The resulting criterion weights and rankings will establish the reference values for subsequent tests.

**AI Evaluation:** The AI interpretation will be examined to determine whether it correctly identifies the uniformity of stakeholder preferences and distinguishes agreement on criterion importance from agreement on the preferred policy alternative. Equal criterion weights do not imply equal alternative performance or equal TOPSIS scores.

#### 4.3.2 Scenario S2: Extreme Preference Dominance

**Objective:** Evaluate how the MCDM engine processes extreme pairwise comparison judgments and identifies potentially inconsistent preferences.

This scenario will examine two configurations representing the maximum and minimum permitted comparison values under Saaty's conventional 1–9 AHP intensity scale.

**Configuration A — Maximum Preference**

Every participant will select the maximum permitted preference intensity for each pairwise comparison:

$$
a_{ij}=9,\qquad i<j
$$

**Configuration B — Minimum Preference**

Every participant will select the minimum permitted preference value for each pairwise comparison:

$$
a_{ij}=\frac{1}{9},\qquad i<j
$$

For both configurations, the diagonal entries will be assigned a value of 1, and the remaining matrix entries will be calculated using the reciprocal property:

$$
a_{ji}=\frac{1}{a_{ij}}
$$

These configurations will be used to examine how extreme judgments affect the resulting AHP criterion weights, consistency ratios, and policy alternative rankings.

Extreme preference values do not necessarily produce internally consistent judgments. Assigning the maximum or minimum value to every unique comparison can introduce contradictions between the implied relative importance of different criteria.

The MCDM engine will therefore be evaluated on its ability to construct reciprocal comparison matrices correctly, calculate the resulting consistency ratios, and flag inconsistent judgments when the configured threshold is exceeded.

In accordance with PoliInsight's existing behavior, inconsistent responses will remain available for subsequent processing but will be flagged.

**AI Evaluation:** The AI interpretation will be evaluated on whether it distinguishes strong preferences from internally consistent preferences. In particular, it should recognize when extreme judgments produce inconsistencies and communicate those limitations rather than presenting the resulting preference structure as unambiguous.

### 4.4 Stakeholder Heterogeneity and Influence Testing

The next three scenarios examine how differences in stakeholder preferences, voting power, and participation affect preference aggregation and the resulting policy rankings.

#### 4.4.1 Scenario S3: Minority Stakeholder Breakaway

**Objective:** Determine how changes in the preferences of a single stakeholder group affect aggregate criterion weights and policy alternative rankings.

The experiment begins with the uniform preference configuration established in Scenario S1.

The preferences of one stakeholder group are then modified while the other four groups retain their original preferences.

For this experiment, the Environmental Advocates group will be assigned a stronger preference for Environmental Benefit. All stakeholder groups will retain equal voting power.

Three levels of preference disagreement will be examined.

| Variation | Preference Configuration |
|---|---|
| S3-A | Mild disagreement |
| S3-B | Moderate disagreement |
| S3-C | Strong disagreement |

The purpose of these variations is to observe how progressively greater differences between one group's preferences and the baseline affect the aggregate criterion weights.

As the Environmental Advocates group's preferences diverge from the baseline, its group-level criterion weights are expected to change. The session-level aggregate weights should also respond to these changes according to the aggregation method and the group's assigned voting power.

However, changes in criterion weights do not necessarily produce changes in policy rankings.

The evaluation will therefore examine changes in aggregate criterion weights, stakeholder preference divergence, the identity of the highest-ranked alternative, and ranking stability using Spearman's rank correlation coefficient and Kendall's rank correlation coefficient.

**AI Evaluation:** The AI interpretation will be assessed on whether it correctly identifies the stakeholder group responsible for the preference change, describes the resulting differences in criterion weights, and accurately attributes any changes in policy rankings to the modified preferences.

#### 4.4.2 Scenario S4: Competing Stakeholder Priorities

**Objective:** Evaluate how PoliInsight aggregates heterogeneous stakeholder preferences when different groups prioritize different evaluation criteria.

Unlike Scenario S3, which modifies the preferences of a single stakeholder group, this scenario assigns distinct preference profiles to all five stakeholder groups.

**Table 3. Stakeholder Criterion-Weight Profiles**

| Stakeholder Group | Implementation Cost | Accessibility Improvement | Environmental Benefit | Implementation Time |
|---|---:|---:|---:|---:|
| Community Residents | 0.15 | 0.55 | 0.15 | 0.15 |
| Transportation Department | 0.15 | 0.35 | 0.15 | 0.35 |
| City Government | 0.45 | 0.25 | 0.15 | 0.15 |
| Environmental Advocates | 0.10 | 0.15 | 0.65 | 0.10 |
| Business Community | 0.40 | 0.30 | 0.10 | 0.20 |

Each stakeholder group emphasizes different aspects of the transportation policy scenario.

Community Residents place the greatest importance on accessibility, while the Transportation Department places equal emphasis on accessibility improvement and implementation time.

City Government and the Business Community place greater importance on implementation cost. Environmental Advocates prioritize environmental benefit.

These preference profiles provide a controlled representation of competing stakeholder priorities.

The MCDM engine will be evaluated on its ability to aggregate the different stakeholder judgments while accounting for the specified stakeholder voting powers.

The resulting group-level and session-level criterion weights will be examined, along with the alternative rankings associated with the different preference configurations.

Different stakeholder groups may produce different TOPSIS rankings because their preferred criteria differ.

**AI Evaluation:** The AI interpretation will be assessed on its ability to explain the differences among stakeholder priorities and describe how these preferences contribute to the aggregate evaluation. It should distinguish the collective mathematical result from unanimous stakeholder agreement and avoid treating one group's preferences as inherently more appropriate than another's.

#### 4.4.3 Scenario S5: Stakeholder Voting Power and Participation Imbalance

**Objective:** Determine how stakeholder voting power and differences in participation influence the aggregate MCDM results.

This scenario evaluates whether PoliInsight appropriately distinguishes between the number of participants representing a stakeholder group and the voting power assigned to that group.

Three configurations will be examined.

**Configuration A — Equal Voting Power**

Five stakeholder groups are assigned equal voting power:

$$
\mathbf{V}=[0.20,0.20,0.20,0.20,0.20]
$$

Each stakeholder group contains one participant.

**Configuration B — Unequal Voting Power**

The stakeholder voting powers are modified to:

$$
\mathbf{V}=[0.10,0.10,0.20,0.40,0.20]
$$

The Environmental Advocates group receives the greatest voting power at 40%.

Participant preferences remain unchanged to isolate the effects of the voting-power adjustment.

**Configuration C — Participation Imbalance**

Stakeholder voting power is returned to an equal distribution, while the number of participants within each group is modified.

| Stakeholder Group | Participants | Voting Power |
|---|---:|---:|
| Community Residents | 5 | 20% |
| Transportation Department | 5 | 20% |
| City Government | 5 | 20% |
| Environmental Advocates | 30 | 20% |
| Business Community | 5 | 20% |

In this configuration, Environmental Advocates account for 60% of participant submissions but retain only 20% of the assigned stakeholder voting power.

To isolate the effect of participation differences, all participants within a stakeholder group will initially submit identical pairwise comparison judgments.

Under PoliInsight's grouped aggregation model, increasing the number of participants within a group should not automatically increase that group's assigned voting power.

When the within-group aggregate preferences remain unchanged, modifying participant counts alone should not affect the final session aggregate matrix.

In contrast, changing the configured voting powers alters the relative contribution of the stakeholder groups to the session-level aggregation.

The evaluation will compare the resulting aggregate criterion weights and alternative rankings across the three configurations to determine whether the system correctly applies these distinctions.

**AI Evaluation:** The AI interpretation will be assessed on its ability to distinguish stakeholder participation from voting power. In particular, it should recognize that a group representing the majority of participant submissions does not necessarily possess majority voting power under the grouped aggregation method.

### 4.5 Ranking Stability and Sensitivity Testing

Scenarios S6 through S9 examine how changes to the decision matrix, evaluation criteria, and criterion weights affect the stability of TOPSIS rankings.

These experiments focus on the sensitivity of the ranking process to changes in the underlying decision problem.

#### 4.5.1 Scenario S6: Removal of the Winning Alternative

**Objective:** Determine how removing the highest-ranked alternative affects the relative rankings of the remaining policy alternatives.

The experiment begins by executing the baseline scenario and identifying the highest-ranked alternative.

The winning alternative is then removed from the decision matrix, and TOPSIS is recalculated using the original criterion weights.

Keeping the criterion weights unchanged isolates the effect of modifying the available alternatives.

The new rankings are compared with the original rankings to identify changes in the relative ordering of the remaining alternatives.

Because TOPSIS normalization and ideal solutions depend on the available alternatives, removing one alternative may affect the closeness coefficients and relative rankings of the remaining alternatives.

An illustrative ranking change is shown below.

| Alternative | Original Rank | Rank After Removal |
|---|---:|---:|
| A1 | 1 | Removed |
| A2 | 2 | 2 |
| A3 | 3 | 1 |
| A4 | 4 | 3 |
| A5 | 5 | 4 |

This example illustrates a possible rank reversal between A2 and A3.

The example is not a calculated result of the synthetic transportation scenario.

The evaluation will measure the number of rank reversals among surviving alternatives, changes in TOPSIS closeness coefficients, changes in the highest-ranked alternative, and rank correlations using Spearman's rho and Kendall's tau.

Rank correlations will be calculated using only the alternatives present in both configurations.

**AI Evaluation:** The AI interpretation will be assessed on its ability to distinguish changes in numerical rank positions from changes in relative ordering. It should also recognize the role of normalization and the positive and negative ideal solutions in explaining changes to the remaining alternatives' TOPSIS scores.

#### 4.5.2 Scenario S7: Removal of an Evaluation Criterion

**Objective:** Determine how removing an evaluation criterion affects the resulting policy alternative rankings.

The experiment will begin with the four-criterion baseline configuration. Each evaluation criterion will then be removed individually, producing four experimental variations.

| Variation | Removed Criterion |
|---|---|
| S7-A | Implementation Cost |
| S7-B | Accessibility Improvement |
| S7-C | Environmental Benefit |
| S7-D | Implementation Time |

Following the removal of a criterion, the remaining criterion weights will be **renormalized** so that their sum remains equal to 1.

Let $\mathbf{w}$ represent the original criterion-weight vector and $R$ represent the set of retained evaluation criteria.

The renormalized weight for each retained criterion will be calculated as:

$$
\boxed{
w_i'=\frac{w_i}{\sum_{j\in R}w_j},
\qquad i\in R
}
$$

Where:

- $w_i'$ represents the renormalized weight of criterion $i$.
- $w_i$ represents its original AHP-derived weight.
- $R$ represents the set of criteria retained after removal.

For example, consider the baseline criterion weights:

$$
\mathbf{W}=[0.25,0.25,0.25,0.25]
$$

If Implementation Cost is removed, the remaining weights will initially be:

$$
\mathbf{W}_{R}=[0.25,0.25,0.25]
$$

Their sum is:

$$
\sum_{j\in R}w_j=0.75
$$

After renormalization:

$$
\mathbf{W}'=
\left[
\frac{0.25}{0.75},
\frac{0.25}{0.75},
\frac{0.25}{0.75}
\right]
$$

Therefore:

$$
\boxed{
\mathbf{W}'=
\left[
\frac{1}{3},
\frac{1}{3},
\frac{1}{3}
\right]
}
$$

The renormalized weights will be used to recalculate the TOPSIS closeness coefficients and alternative rankings using the reduced decision matrix.

The original AHP judgments will not be recollected or recalculated for this experiment. This allows the evaluation to isolate the effect of removing an evaluation criterion while preserving the relative importance relationships among the remaining criteria.

The resulting alternative rankings will be compared against the baseline to determine how the exclusion of each criterion affects the decision outcome.

**AI Evaluation:** The AI interpretation will be evaluated on whether it correctly identifies the removed criterion, explains the effects of renormalization, and accurately describes changes in the relative rankings of the remaining policy alternatives.

#### 4.5.3 Scenario S8: Gradual Criterion Weight Perturbation

**Objective:** Evaluate the sensitivity of TOPSIS rankings to gradual changes in criterion importance.

This scenario begins with equal criterion weights:

$$
\mathbf{W}=[0.25,0.25,0.25,0.25]
$$

The weight assigned to one criterion is progressively increased while the weights assigned to the remaining three criteria are reduced proportionally.

The following configuration illustrates the process using Implementation Cost as the modified criterion.

**Table 4. Criterion Weight Perturbation Configurations**

| Test | Implementation Cost | Accessibility Improvement | Environmental Benefit | Implementation Time |
|---|---:|---:|---:|---:|
| W0 | 0.25 | 0.25 | 0.25 | 0.25 |
| W1 | 0.40 | 0.20 | 0.20 | 0.20 |
| W2 | 0.55 | 0.15 | 0.15 | 0.15 |
| W3 | 0.70 | 0.10 | 0.10 | 0.10 |
| W4 | 0.85 | 0.05 | 0.05 | 0.05 |

The experiment will be repeated for each of the four evaluation criteria.

The purpose is to identify how changes in criterion weights affect the TOPSIS closeness coefficients and the relative ordering of policy alternatives.

Rankings may remain unchanged across some weight configurations while changing at particular weight thresholds.

These transitions are relevant for determining the sensitivity of the policy evaluation to changes in stakeholder priorities.

For example, increasing the importance of implementation cost may favor lower-cost alternatives, while increasing the importance of accessibility may favor alternatives with greater accessibility benefits.

The actual ranking transitions will be determined by the experiment.

The evaluation will examine the criterion-weight thresholds associated with changes in the winning alternative, the number of ranking transitions, changes in TOPSIS closeness coefficients, Spearman and Kendall rank correlations, and the stability of the highest-ranked alternative across the tested configurations.

**AI Evaluation:** The AI interpretation will be assessed on its ability to explain how changes in criterion importance influence policy rankings. It should identify relevant ranking transition thresholds and distinguish alternatives whose rankings remain stable from those whose rankings are sensitive to changes in criterion weights.

#### 4.5.4 Scenario S9: Addition and Removal of a Dominated Alternative

**Objective:** Determine whether introducing an inferior policy alternative affects the relative rankings of existing alternatives.

This experiment extends the original five-alternative decision matrix by introducing a sixth alternative.

The additional alternative is constructed to perform worse than an existing alternative across all four evaluation criteria.

**Table 5. Additional Dominated Policy Alternative**

| ID | Alternative | Implementation Cost ($M) ↓ | Accessibility Improvement ↑ | Environmental Benefit ↑ | Implementation Time (Months) ↓ |
|---|---|---:|---:|---:|---:|
| A6 | Inefficient Infrastructure Program | 9.0 | 40 | 30 | 30 |

Alternative A6 is dominated by Pedestrian Accessibility (A4), which has a lower implementation cost, greater accessibility improvement, greater environmental benefit, and a shorter implementation time.

The experiment consists of the following sequence:

1. Execute TOPSIS using the original alternatives A1–A5.
2. Add A6 to the decision matrix.
3. Recalculate TOPSIS without modifying the criterion weights.
4. Compare the rankings of A1–A5 before and after introducing A6.
5. Remove A6 and verify whether the original results are restored.

Under conventional TOPSIS, with correctly specified criterion directions and positive criterion weights, A6 should not outrank A4.

However, introducing A6 may affect the normalization process, ideal solutions, and closeness coefficients of other alternatives.

Consequently, the relative rankings of the original alternatives may change even though their performance values and criterion weights remain unchanged.

The evaluation will determine whether such changes occur under PoliInsight's TOPSIS implementation.

**AI Evaluation:** The AI interpretation will be assessed on whether it correctly recognizes that A6 is dominated by A4 and accurately explains any changes in the rankings of the original alternatives. It should distinguish changes caused by the composition of the decision matrix from actual changes in the performance of existing alternatives.

### 4.6 Judgment Consistency and Reliability Testing

The final scenario evaluates how PoliInsight handles contradictory participant preferences.

#### 4.6.1 Scenario S10: Inconsistent Stakeholder Judgments

**Objective:** Evaluate whether PoliInsight correctly identifies and flags inconsistent AHP pairwise comparison judgments at different levels of stakeholder participation and aggregation.

This scenario will introduce deliberate inconsistencies into participant preference matrices.

For example, a participant may express the following relative importance judgments:

$$
C1>C2
$$

$$
C2>C3
$$

$$
C3>C1
$$

These judgments form a contradictory preference cycle.

The corresponding numerical comparisons will be:

$$
a_{12}=5,\qquad a_{23}=5,\qquad a_{13}=\frac{1}{5}
$$

The comparisons involving the fourth criterion will be assigned a value of 1, and the remaining entries will be constructed using the reciprocal property.

Three experimental variations will be evaluated.

| Variation | Configuration |
|---|---|
| S10-A | One participant submits an inconsistent comparison matrix. |
| S10-B | Every participant within one stakeholder group submits an inconsistent comparison matrix. |
| S10-C | All stakeholder groups submit inconsistent comparison matrices. |

PoliInsight's current approach to inconsistent judgments is to **retain the submitted responses while flagging detected inconsistencies**.

Accordingly, the purpose of this experiment will not be to determine whether inconsistent responses are rejected or removed. Instead, the evaluation will examine whether the system correctly calculates consistency ratios, identifies judgments that exceed the configured consistency threshold, and preserves the corresponding inconsistency information during processing.

The test will also examine the resulting stakeholder group and session-level aggregate matrices to determine how inconsistent participant judgments affect the aggregate criterion weights and policy rankings.

The evaluation will compare PoliInsight's consistency calculations and resulting outputs against the independent computational implementation.

**AI Evaluation:** The AI interpretation will be evaluated on whether it recognizes that the results contain flagged inconsistent judgments and explains the potential implications for the reliability of the evaluation.

The AI should distinguish between the mathematical validity of the calculations and the consistency of the underlying stakeholder judgments.

### 4.7 Evaluation and Comparison Methodology

The deterministic evaluation will be conducted using two separate implementations of the MCDM processing procedures: the PoliInsight application and an independent computational testing implementation.

Both implementations will process equivalent experimental inputs so that their outputs can be compared.

The purpose of this comparison is to determine whether PoliInsight correctly implements the mathematical procedures used for preference matrix construction, stakeholder aggregation, AHP weighting, consistency analysis, and TOPSIS ranking.

#### 4.7.1 Independent Computational Testing Implementation

An independent testing implementation will be developed specifically for the system evaluation.

Unlike PoliInsight, this implementation will not be constrained by the application's web interface, session management functionality, or existing processing architecture.

Its purpose will be limited to executing the mathematical procedures required by the experiments and producing reference results for comparison.

The testing implementation will be developed separately from PoliInsight's existing calculation functions. This separation is intended to reduce the possibility that implementation-specific errors are reproduced in both systems.

It will also allow the calculations and intermediate results to be examined independently of the application's broader functionality.

The implementation will reproduce the relevant deterministic processing stages, including:

1. Construction of participant pairwise comparison matrices.
2. Within-group aggregation using the geometric mean.
3. Across-group aggregation using the weighted geometric mean and stakeholder voting powers.
4. AHP criterion-weight calculation.
5. AHP consistency analysis.
6. TOPSIS normalization, closeness coefficient calculation, and alternative ranking.

The independent implementation will use the same mathematical definitions and experimental inputs as PoliInsight.

Where the experiments introduce modifications to participant preferences, voting power, criteria, or alternatives, those modifications will be applied consistently to both implementations.

The independent results will serve as reference values for identifying discrepancies in PoliInsight's deterministic processing.

Although developing the implementations separately can reduce shared implementation errors, agreement between them will not, by itself, establish that the algorithms are mathematically correct. Their calculations must also conform to the specified AHP, aggregation, and TOPSIS procedures.

#### 4.7.2 Comparison of Deterministic Results

For each experiment, PoliInsight and the independent testing implementation will process the same synthetic policy data and participant preference configurations.

The outputs from both implementations will be compared at the relevant stages of the deterministic pipeline.

**Table 6. Deterministic Processing Comparison**

| Processing Component | PoliInsight Output | Independent Reference Output | Comparison Objective |
|---|---|---|---|
| Participant Preference Matrices | Application-generated comparison matrices | Independently constructed matrices | Verify preference mapping and reciprocal relationships. |
| Within-Group Aggregation | Stakeholder group matrices | Independently calculated geometric means | Verify participant preference aggregation. |
| Across-Group Aggregation | Session aggregate matrix | Independently calculated weighted geometric mean | Verify the application of stakeholder voting power. |
| AHP Criterion Weights | Application-generated criterion weights | Independently calculated AHP weights | Verify the weighting procedure. |
| AHP Consistency Analysis | Consistency ratios and inconsistency flags | Independently calculated consistency ratios | Verify consistency calculations and detection. |
| TOPSIS Closeness Coefficients | Application-generated closeness coefficients | Independently calculated TOPSIS coefficients | Verify alternative evaluation calculations. |
| TOPSIS Rankings | Application-generated alternative ordering | Independently calculated alternative ordering | Verify ranking correctness. |

Comparing intermediate calculations will allow discrepancies to be associated with specific processing stages rather than relying exclusively on differences in the final alternative rankings.

For example, an incorrect ranking may originate from an incorrectly constructed participant matrix, an error in stakeholder aggregation, an incorrect criterion-weight calculation, or the TOPSIS ranking procedure itself.

This approach will provide a structured means of examining the deterministic behavior of PoliInsight under the different test conditions.

#### 4.7.3 Comparison Across Test Scenarios

In addition to comparing PoliInsight with the independent testing implementation, the evaluation will examine how the deterministic results change across the ten test scenarios.

Scenario S1 will establish the baseline configuration using uniform stakeholder preferences and equal voting power.

Subsequent scenarios will introduce controlled modifications to stakeholder preferences, voting power, participant counts, evaluation criteria, and available policy alternatives.

The resulting criterion weights, TOPSIS closeness coefficients, and rankings will be compared with the appropriate reference configurations.

For experiments involving changes in alternative rankings, Spearman's rank correlation coefficient and Kendall's rank correlation coefficient will be used to evaluate ranking similarity.

Changes in the highest-ranked alternative and changes in criterion weights and TOPSIS closeness coefficients will also be examined where applicable.

The purpose of these comparisons is to determine how the deterministic MCDM engine responds to different decision-making conditions and whether the observed behavior is consistent with the underlying mathematical procedures.

#### 4.7.4 AI Interpretation Evaluation

The deterministic results generated during the experiments will also provide the basis for evaluating PoliInsight's AI interpretation component.

Each test scenario will introduce specific conditions that the AI should recognize when analyzing the policy evaluation.

The evaluation will examine whether the AI accurately describes the policy scenario, identifies relevant stakeholder preferences and disagreements, and explains the relationships between changes in experimental inputs and the resulting alternative rankings.

The evaluation will also examine whether the AI recognizes limitations associated with inconsistent judgments, ranking sensitivity, and changes to the decision matrix.

The AI-generated explanations will be assessed against the deterministic results and the documented experimental conditions.

This will allow the evaluation to distinguish between the correctness of the underlying MCDM calculations and the accuracy of the AI's interpretation of those calculations.

### 4.8 Evaluation Objectives

The proposed system evaluation will address three principal objectives.

**Mathematical Correctness:** Determine whether PoliInsight's deterministic MCDM engine produces results consistent with an independently developed computational implementation of the specified mathematical procedures.

**Behavioral Stability:** Examine how changes in stakeholder preferences, voting power, participation, evaluation criteria, and policy alternatives affect the resulting aggregate criterion weights and TOPSIS rankings.

**Interpretation Accuracy:** Evaluate whether PoliInsight's AI interpretation component accurately represents the policy scenario, stakeholder relationships, deterministic results, and limitations associated with the different experimental conditions.

The ten test scenarios will provide a controlled and reproducible framework for examining these objectives.

Following approval of the proposed methodology, the independent testing implementation will be developed, the experiments will be conducted, and the resulting observations will be documented and analyzed.
