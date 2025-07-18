<div style="font-family: Georgia, serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px;">

# Ameliorating Performance of Random Forest using Data Clustering

<div align="center" style="margin-bottom: 30px;">
  <h3 style="color: #2c3e50; font-weight: normal; margin-bottom: 20px;">Authors</h3>
  <p style="font-size: 16px; color: #34495e;">
    <a href="mailto:umuna201429@bscse.uiu.ac.bd" style="color: #2980b9; text-decoration: none;">Ummay Maria Muna</a> • 
    <a href="mailto:sbiswas201418@bscse.uiu.ac.bd" style="color: #2980b9; text-decoration: none;">Shanta Biswas</a> • 
    <a href="mailto:szarif202009@bscse.uiu.ac.bd" style="color: #2980b9; text-decoration: none;">Syed Abu Ammar Muhammad Zarif</a> • 
    <a href="mailto:dewanfarid@cse.uiu.ac.bd" style="color: #2980b9; text-decoration: none;">Dewan Md. Farid</a>
  </p>
</div>

---

## Abstract

Random Forest is one of the most popular supervised learning ensemble methods in machine learning. Random Forest engenders a set of random trees and considers majority voting technique to classify known and unknown data instances. In Random Forest, decision tree induction is used as a baseline classifier. 

Decision tree is a top-down divide and conquer recursive algorithm that applies feature selection technique to select the root/best feature, including:

- **ID3** (Iterative Dichotomiser 3)
- **C4.5** (an extension of ID3) 
- **CART** (Classification and Regression Tree)

<div style="background-color: #f8f9fa; padding: 20px; border-left: 4px solid #2980b9; margin: 20px 0;">
<strong>Key Contribution:</strong> In this paper, we have proposed a new approach to improve the performance of Random Forest classifier using clustering technique. This proposed idea can be applied for Big Data mining.
</div>

---

## Methodology

Our approach follows a systematic two-stage process:

**Stage 1: Data Clustering**  
We cluster the data into several clusters using K-Means Clustering algorithm to create homogeneous subgroups within the dataset.

**Stage 2: Ensemble Classification**  
We apply the Random Forest technique independently to each cluster, leveraging the reduced complexity and improved data homogeneity within each cluster.

---

## System Architecture

<div align="center" style="margin: 30px 0;">
  <img src="./rfwoc/proposed_system_diagram.jpeg" 
       alt="Proposed System Architecture" 
       style="width: 70%; height: auto; border: 1px solid #bdc3c7; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />
  <br>
  <p style="font-style: italic; color: #7f8c8d; margin-top: 10px; font-size: 14px;">
    Figure 1: Proposed System Architecture
  </p>
</div>

---

## Experimental Results

We have conducted comprehensive experiments comparing our proposed clustering-based Random Forest approach with the traditional Random Forest algorithm. The evaluation was performed on five benchmark datasets obtained from the UCI Machine Learning Repository.

<div style="background-color: #ecf0f1; padding: 20px; border-radius: 4px; margin: 20px 0;">
<table style="width: 100%; border-collapse: collapse;">
  <tr>
    <td style="text-align: center; padding: 15px; border-right: 1px solid #bdc3c7;">
      <strong>Enhanced Performance</strong><br>
      <span style="color: #27ae60;">Improved accuracy over traditional RF</span>
    </td>
    <td style="text-align: center; padding: 15px; border-right: 1px solid #bdc3c7;">
      <strong>Comprehensive Evaluation</strong><br>
      <span style="color: #27ae60;">Tested on 5 UCI datasets</span>
    </td>
    <td style="text-align: center; padding: 15px;">
      <strong>Scalability</strong><br>
      <span style="color: #27ae60;">Suitable for Big Data applications</span>
    </td>
  </tr>
</table>
</div>

**Key Findings:** Our proposed Random Forest technique demonstrates superior performance compared to the traditional Random Forest algorithm across all evaluated datasets, showing particular promise for large-scale data mining applications.

---

## Publication Details

<div align="center" style="margin: 30px 0;">
  <a href="https://ieeexplore.ieee.org/document/10441376" 
     style="display: inline-block; 
            background-color: #34495e; 
            color: white; 
            padding: 12px 30px; 
            text-decoration: none; 
            border-radius: 4px; 
            font-weight: bold; 
            font-size: 16px;
            transition: background-color 0.3s ease;">
    Access Full Paper
  </a>
</div>

<div align="center" style="margin-top: 20px;">
  <p style="font-style: italic; color: #7f8c8d; font-size: 14px;">
    Published in IEEE Conference Proceedings
  </p>
</div>

</div>
