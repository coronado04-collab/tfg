# Summaries of Key Articles BSc Thesis

## Part I: Hickl et al. (2025)

### 1\. Article Identification

**Title:**  Segmentation of dense and multi-species bacterial colonies using models trained on synthetic microscopy images  
**Authors:**  Vincent Hickl, Abid Khan, René M. Rossi, Bruno F. B. Silva, Katharina Maniura-Weber  
**Year:**  2025  
**Journal / Conference:**  PLOS Computational Biology

### 2\. Main Themes and Ideas

Article’s core research objective: to overcome the data-annotation bottleneck in bacterial image segmentation by developing a novel synthetic data generation pipeline.

* **Primary Research Objective:**  The main goal is to develop a highly adaptable method for creating accurate single-cell segmentation models for bacterial colonies, crucially without the need for tedious and time-consuming manual annotation of training images.  
* **Problem Being Addressed:**  The central challenge the paper tackles is the significant bottleneck caused by the lack of high-quality, large-scale annotated training datasets for segmenting bacteria. This problem is especially acute when dealing with dense, multi-species colonies or when analyzing images captured under suboptimal, yet clinically relevant, conditions.  
* **Problem's Importance:**  The significance of this challenge is multifaceted, with direct implications for several critical areas of research and clinical practice:  
  * **Antibiotic Resistance:**  The ability to analyze bacterial self-organization at the single-cell level is crucial for understanding the mechanics of biofilm architecture. This knowledge is essential for developing novel antimicrobial treatments designed to disrupt biofilm formation, a key mechanism of antibiotic resistance.  
  * **Medical Imaging:**  The study addresses the difficulty of analyzing images from clinically relevant surfaces, such as PDMS (polydimethylsiloxane), which are not optimized for high signal-to-noise ratios. This contrasts with many prior studies that rely on ideal imaging conditions (e.g., glass coverslips) that do not reflect complex clinical environments.  
  * **Automated Diagnostics:**  By overcoming the data annotation barrier, the methods developed in this paper promise to facilitate the quantitative analysis of bacterial infections. This could pave the way for the development of rapid and automated diagnostic tools suitable for clinical settings.

### 3\. Methodology Breakdown

#### 3.1 Segmentation

* **Type of Segmentation:**  The authors employ a deep learning-based approach. The methodology features two core components:  
  * A  **cycle Generative Adversarial Network (cycleGAN)**  is used to translate simple, "raw" synthetic images into realistic-looking microscopy images that can serve as training data. First, real microscop images of bacteria are recorded with different microscope techniques. Then, custom computational models are used to create ‘raw’ synthetic images of bacteria – images in which cell densities and shapes are approximately equal to those in the real images but that do not contain noise. The real and raw synthetic images are then used as inputs for a cycleGAN, used here to ‘process’ synthetic images by giving them optical characteristics to resemble the real images. Together with the original masks of the raw synthetic images, these processed synthetic images are then used to train neural networks to perform single-cell segmentation and species classification on real images.  
  * The  **Omnipose**  package is used to train the final deep neural network segmentation model on this processed synthetic dataset.  
* **Input Data Characteristics:**  The experiments were conducted using two bacterial species,  ***Pseudomonas aeruginosa***  (rod-shaped) and  ***Staphylococcus aureus***  (spherical). Images were captured using both  **onfocal laser scanning**  and  **brightfield microscopy**. The bacteria were cultured on challenging  **PDMS films**  to deliberately simulate suboptimal imaging conditions relevant to clinical research.  
* **Labels / Ground Truth Generation:**  The ground truth label masks required for training the segmentation model were generated computationally at the same time as the raw synthetic images. This completely circumvents the need for any manual human annotation, which is the central innovation of the pipeline.  
* **Loss Functions and Evaluation Metrics:**  The article does not specify the loss functions used to train the Omnipose segmentation model. However, it provides a clear set of metrics for evaluating segmentation performance:  
  * **Panoptic Quality (PQ):**  A comprehensive metric combining segmentation and recognition quality.  
  * **Intersection over Union (IoU):** measures the pixel-level overlap between predicted and ground truth cell masks.  
  * **F1 score:** measures the model's ability to correctly identify the presence of cells.  
* **Strengths and Limitations:**  
  * **Strengths:**  The primary strength of the approach is its high adaptability. It can be tailored to different imaging modalities (confocal, brightfield) and experimental conditions with relative ease. It achieves accurate segmentation even in dense colonies without the significant labor cost of manual labeling.  
  * **Limitations:**  An implicit limitation is that the method was demonstrated on only two bacterial species with very distinct morphologies (rods and spheres). Furthermore, the scope is limited to cell segmentation; it does not address the segmentation of other critical features for antibiotic susceptibility testing, such as  **inhibition zones** .  
* **Relevance for** **Thesis:**  The segmentation methodology is highly relevant. The cycleGAN-based synthetic data generation pipeline is a very promising technique for creating a large, high-quality training dataset for segmenting  *Klebsiella*  colonies and their corresponding inhibition zones from Petri dish images. Adopting this method could dramatically reduce or even eliminate the need for manual annotation. The immediate practical application for the thesis is to create a small set of 'raw synthetic' images modeling  *Klebsiella*  colonies and inhibition zones, and a corresponding set of real Petri dish images, which would serve as the two domains for training a new cycleGAN from scratch.

#### 3.2 Classification

* **Classification Objective:**  The objective is a binary classification task: distinguishing between the rod-shaped  *P. aeruginosa*  and the spherical  *S. aureus*  in mixed-colony images.  
* **Model Architectures Used:**  The system cleverly performs simultaneous segmentation and classification by combining two separate Omnipose models. One model is trained specifically to segment rods, and the other is trained to segment circles. A cell is classified based on which model successfully segments it.  
* **Input Features:**  The models operate directly on the raw microscopy images to identify and segment cells. The classification is an emergent property of the specialized segmentation models.  
* **Training Strategy:**  Both models were trained exclusively on the "processed synthetic images" generated by the cycleGAN. The paper specifies the dataset sizes used for the mixed-colony experiments: 226 images for the confocal microscopy model and 441 for the brightfield model.  
* **Evaluation Metrics:** The paper evaluates the combined segmentation and identification performance using Panoptic Quality (PQ).  
* **Key Results:**  The approach was effective in distinguishing between the two species based on their distinct morphologies in both confocal and brightfield images.  
* **Relevance for**  **Thesis:**  This classification approach is conceptually relevant to the thesis goal of antibiotic resistance prediction. Although this paper classifies species by morphology, the underlying principle can be adapted. After segmenting  *Klebsiella*  colonies and their associated inhibition zones, a dedicated classification model, such as a Convolutional Neural Network (CNN), could be trained. The input to this classifier would not be the entire image, but rather the segmented image patches of individual  *Klebsiella*  colonies and their corresponding inhibition zones, from which it would learn to predict resistance based on textural and morphological features.

### 4\. Evidence and Results

This section summarizes the key quantitative results reported by Hickl et al. to validate their synthetic data approach against a pre-trained baseline model.

* **Performance Improvements:**  The segmentation model trained on the synthetically generated data (the "synthetic model") was shown to significantly outperform an existing, pre-trained model from the literature ('Bact\_fluor\_omni') when tasked with segmenting dense  *P. aeruginosa*  monolayers.  
* **Quantitative Metrics:**  The synthetic model achieved a  **Panoptic Quality (PQ) of 0.67** , a marked improvement over the 0.55 PQ of the pre-trained 'Bact\_fluor\_omni' model. Furthermore, it correctly identified  **97% of cells** , compared to the baseline's 84%.  
* **Segmentation Quality:**  The model's superiority was particularly pronounced in the "segmentation quality" component of PQ, which is measured by the Intersection over Union (IoU). This indicates that the model trained on synthetic data was able to delineate the precise boundaries of individual cells more accurately.  
* **Multi-species Performance:**  The methodology was also successfully applied to segment and classify multi-species colonies imaged with both confocal and brightfield microscopy, clearly demonstrating its adaptability to different experimental setups.

### 5\. Conclusions of the Article

The authors successfully developed and validated a method to train accurate, single-cell segmentation models using only synthetic microscopy images. By processing these synthetic images with a cycleGAN, they created a realistic training set that eliminates the need for any manual annotation.

* **Contributions:**   
  * A simple, adaptable, and accessible computational tool for creating bespoke segmentation models tailored to specific experimental systems.  
  * The achievement of accurate single-cell segmentation of both dense and multi-species colonies, even under suboptimal imaging conditions that mimic clinical scenarios.  
  * A demonstration of the method's applicability to two common microscopy techniques: brightfield and confocal laser scanning microscopy.  
* **Implications:**  The authors suggest that their method promises to greatly simplify the quantitative description of bacterial infections. As a practical application, this framework could be used to develop rapid diagnostic tools for deployment in clinical settings.

### 6\. Limitations and Open Challenges

A critical analysis of the study's constraints and unanswered questions reveals key opportunities for future research, which are outlined below.

* **Dataset Limitations:**  The models were developed and tested using only two bacterial species,  *P. aeruginosa*  and  *S. aureus* , which have very distinct and simple morphologies (rods and spheres). The generalizability of this approach to other species with more complex or variable shapes, such as  *Klebsiella* , has not been demonstrated.  
* **Scope of Application:**  The paper focuses exclusively on single-cell segmentation and morphological species identification. It does not address other critical features relevant to a clinical microbiology context, most notably the analysis of  **inhibition zones** , which are fundamental for antibiotic susceptibility testing.  
* **Synthetic Data Simplicity:**  The "raw" synthetic images are based on simple geometric primitives (rectangles with caps and disks). While this was effective for the chosen species, this approach may not be sufficient to capture the more subtle textural and morphological variations required for more complex classification tasks, such as predicting antibiotic resistance from colony appearance.

### 7\. Relevance to My BSc Thesis

* **Which part is most relevant?**  
  **Segmentation:**  The core segmentation methodology is  **highly relevant** .  
    
* **What ideas, methods, or architectures could I reuse or adapt?**  
  **Synthetic Data Generation:**  The central idea of using a  **cycleGAN**  to translate simple, programmatically generated images into a realistic training dataset is the single most valuable and adaptable method from this paper. This technique could be applied to generate synthetic images of  *Klebsiella*  colonies and inhibition zones, potentially eliminating the need for a large, manually annotated dataset.  
  **Segmentation Model:**  The use of the  **Omnipose**  framework as the underlying segmentation model is a concrete architectural choice that could be directly adopted for the segmentation component of the thesis project.  
    
* **What gaps remain that my thesis could address?**  
  **Application to**  ***Klebsiella***  **:**  This paper does not involve  *Klebsiella* . The thesis can directly address this gap by applying and validating the synthetic data generation approach specifically for segmenting  *Klebsiella*  colonies.  
* **Inhibition Zone Segmentation:** The thesis must extend the methodology to not only identify bacterial colonies but also to accurately segment the zones of growth inhibition around antibiotic disks. This is a critical feature for AST that is entirely unaddressed by this paper.  
* **Resistance Prediction:**  The paper's classification task is limited to identifying species based on gross morphology. The thesis will tackle the much more complex challenge of moving beyond segmentation to  **predict antibiotic resistance** , a critical downstream task that this paper does not address.

## Part II: Signoroni et al. (2023)

### 1\. Article Identification

**Title:**  Hierarchical AI enables global interpretation of culture plates in the era of digital microbiology  
**Authors:**  Alberto Signoroni, Alessandro Ferrari, Stefano Lombardi, Mattia Savardi, Stefania Fontana, Karissa Culbreath  
**Year:**  2023  
**Journal / Conference:**  Nature Communications

### 2\. Main Themes and Ideas

Paper's primary objective: to create DeepColony, a comprehensive, hierarchical AI system for automating the entire workflow of clinical microbiology culture plate interpretation.

* **Primary Research Objective:**  The main goal of this work is to create a comprehensive, hierarchical AI system named  **DeepColony**. This system is designed to tackle the entire, complex task of clinical microbiology culture plate interpretation, a workflow that includes colony counting, presumptive pathogen identification, and a final assessment of the plate's clinical significance.  
* **Problem Being Addressed:**  The central challenge the paper addresses is the automation of the visual interpretation of bacterial culture plates. This task is complex, requires significant expertise, is often subjective, and has remained a largely manual process despite the widespread adoption of Full Laboratory Automation (FLA) in clinical labs.  
* **Problem's Importance:**  The automation of this process is highly significant for modern clinical microbiology for several reasons:  
  * **Antibiotic Resistance:**  Correct and timely identification of pathogens is an essential first step in combating infections and addressing the crisis of antimicrobial resistance. The system provides the necessary upstream analysis for critical downstream tasks like antimicrobial susceptibility testing (AST).  
  * **Medical Imaging:**  Modern FLA systems generate massive streams of high-resolution digital plate images. There is a pressing need for advanced digital analysis tools that can process this data efficiently and consistently.  
  * **Automated Diagnostics:**  The DeepColony system is designed to function as a clinical decision support tool. It can standardize the interpretation of culture plates, reduce the workload of skilled microbiologists by handling routine cases, and shorten diagnostic turnaround times, ultimately leading to more responsive and effective patient care.

### 3\. Methodology Breakdown 

#### 3.1 Segmentation

* **Context:**  Segmentation is an implicit but foundational step within the DeepColony system. It is handled at  **Level 0**  of the system's five-level hierarchical architecture.   
* **Type of Segmentation:**  The method is a deep learning-based approach using a Convolutional Neural Network (CNN), which the authors cite from a previous publication (Ferrari et al., 2017). Its purpose is to produce an "enumeration map" that performs several tasks: identifying the locations of isolated colonies, detecting small aggregates of touching colonies, and flagging larger confluent growth areas. This map is then used at  **Level 1**  to select well-isolated "good colonies" for subsequent analysis.  
* **Input Data Characteristics:**  The input data consists of high-resolution digital scans of urine culture plates grown on sheep blood agar. The images were acquired using a WaspLab™ Full Laboratory Automation system.  
    
* **Strengths and Limitations:**  
  * **Strengths:**  The key strength of this segmentation approach is that it is fully integrated into a larger, clinically-focused workflow. It serves a practical purpose by providing the necessary, cleaned input for the more complex downstream tasks of classification and interpretation.  
  * **Limitations:**  The paper does not provide technical details or performance metrics for the segmentation step itself. The authors also note an intrinsic limitation of the overall system: its inability to reliably identify bacterial species within the confluent areas flagged by this initial step.  
* **Relevance for** **Thesis:** The idea of a hierarchical system where an initial segmentation and "good colony" selection step (Levels 0 and 1\) precedes the main classification task is a crucial architectural concept. This workflow of segment-then-classify should be adopted for the thesis project to ensure that the resistance prediction model operates on high-quality, well-defined colony images.

#### 3.2 Classification

* **Context:**  The classification of bacteria for presumptive pathogen identification is the central technical innovation of the DeepColony system, performed across Levels 2 and 3 of the architecture.  
* **Classification Objective:**  The objective is a multi-class classification task designed to identify bacterial species from a comprehensive set of 32 clinically relevant pathogens commonly found in Urinary Tract Infections (UTIs).  
* **Model Architectures Used:**  The system employs a sophisticated two-stage classification architecture:  
  * **Level 2 ("pathogen aware \- similarity agnostic"):**  A  **Convolutional Neural Network (CNN)**  performs an initial presumptive identification on each individual "good colony" image. It outputs a confidence-ranked vector of probable species for that single colony.  
  * **Level 3 ("similarity aware \- pathogen agnostic"):**  A  **Siamese CNN**  is used to learn a powerful similarity metric. This network maps all colonies from a single plate into a low-dimensional embedding space. In this space, colonies are clustered using the Mean-shift algorithm. A final classification is given for each identified cluster by averaging the pIDv’s from level 2 within the cluster. This contextual analysis refines the initial Level 2 predictions, enforcing consistency across visually similar colonies on the same plate.  
* **Input Features:**  The inputs to the classification models are the images of isolated bacterial colonies that were selected at Level 1\.  
* **Training Strategy:**  The models were trained on a large and clinically relevant dataset comprising  **26,213 isolated colony images** . These images were derived from pure flora cultures, and crucially, the ground truth labels for each colony were established by definitive  **MALDI-ToF identification** , a crucial methodological choice that anchors the visual-feature training to a gold-standard molecular identity, ensuring high-quality labels.  
    
  Individual colony interpretation and contextual interpretation provides information for the laboratory to use. However, the laboratory must ultimately synthesize that information to generate a laboratory result based on culture type and laboratory-specific rules. In DeepColony this is achieved at level 4, where information from level 0 (segmentation and enumeration) and level 3 (pIDv) are combined to compute min-max colony counting range for each identified species on the plate

* **Evaluation Metrics:**  The authors evaluate classification performance using  **Top-1, Top-2, and Top-3 accuracy** . The results are presented visually through comprehensive 32x32 and 16x16 confusion matrices.  
* **Key Results:**  The system demonstrated high performance. The Top-1 accuracy for species identification was  **83.4%**  at the single-colony Level 2 and improved significantly to  **90.6%**  after the context-based refinement at Level 3\.  
* **Relevance for** **Thesis:** It provides direct evidence that deep learning models can successfully classify  *Klebsiella pneumoniae*  from culture plate images. Second, the hierarchical classification architecture is a powerful model to adapt. For the thesis, this architecture can be adapted directly:    
  **1\) Level 2 Analog:**  A CNN is trained to predict a resistance class (e.g., Susceptible, Intermediate, Resistant) from a single segmented  *Klebsiella*  colony or inhibition zone.    
  **2\) Level 3 Analog:**  A Siamese CNN is trained to learn a similarity metric based on the  *visual phenotype of resistance* . For example, it would learn to map colonies with fuzzy edges near the inhibition zone close together in the embedding space, while mapping those with sharp, clear growth inhibition further apart. Clustering in this space would enforce a consistent resistance prediction for all colonies on a plate that exhibit a similar phenotypic response to the antibiotic, dramatically increasing the prediction's robustness.

### 4\. Evidence and Results

This section summarizes the key findings from the paper's large-scale validation on over 5,000 clinical urine cultures, demonstrating the system's performance in a real-world setting.

* **Plate-Level Validation:**  The entire end-to-end DeepColony system was validated on a large clinical dataset of  **5,051 urine cultures** , representing a complete workflow from image acquisition to final interpretation.  
* **Human-Machine Agreement:**  The primary result of the study was that the system achieved a  **95.4% overall agreement**  with the final interpretations made by professional laboratory technologists.  
* **Class-Based Performance:**  The agreement rate varied slightly depending on the interpretation class:  
  * **99.2%**  agreement for "no-growth" cultures.  
  * **95.6%**  agreement for "positive" cultures.  
  * **77.1%**  agreement for "contaminated" cultures.  
* **Safety by Design:**  The authors explain that the lower agreement rate for "contaminated" cultures is the result of a deliberate "safety by design" choice. The system is configured to be precautionary, tending to flag borderline cases as positive to minimize the risk of dangerous false negatives. This ensures that ambiguous cases are passed to a human expert for final interpretation.

### 5\. Conclusions of the Article

The authors successfully developed DeepColony, a unique, hierarchical AI framework that can automate the entire workflow of culture plate interpretation, achieving a high degree of accuracy and agreement with human experts in a clinical setting.

* **Contributions:**  
  * A multi-level system that performs species-specific identification and quantitation, which is described as unprecedented in a machine-assisted context.  
  * A novel two-step classification process that combines single-colony analysis with a context-aware refinement step to improve accuracy and consistency.  
  * A flexible framework that can be integrated with laboratory-specific rules (at Level 4\) to provide clinically relevant decision support that aligns with established protocols.  
* **Implications:**  The system has the potential to reduce laboratory workload, increase standardization and traceability, speed up presumptive diagnosis, and ultimately improve patient care by allowing skilled microbiologists to focus their expertise on the most critical and complex cases.

### 6\. Limitations and Open Challenges

* **Confluent Areas:**  The system has an intrinsic inability to reliably identify bacterial species within areas of confluent growth where individual colonies are not discernible.  
* **Limited Scope of Organisms:**  The study was evaluated on urine cultures. The authors note that the models would need to be adapted and retrained to include organisms that are prevalent in other types of clinical cultures.  
* **No Resistance Prediction:**  The system's primary classification task is presumptive species identification. It does not perform antimicrobial susceptibility prediction, which is a critical next step in the clinical microbiology workflow and represents a major area for future development.

### 7\. Relevance to my BSc Thesis

* **Which part is most relevant?**  
  **Classification:**  The hierarchical classification methodology (Levels 2 and 3\) is  **extremely relevant** .  
    
  **Overall Architecture:**  The entire hierarchical concept of the system (segmentation → classification → interpretation) is also highly relevant as a guiding architectural principle.  
    
* **What ideas, methods, or architectures could I reuse or adapt?**  
  **Hierarchical Classification:**  The core architectural idea of a two-step classification—an initial prediction on single colonies followed by a context-aware refinement using a Siamese network—is a powerful and directly adaptable method for antibiotic resistance prediction. For the thesis, this architecture can be adapted directly:    
  **1\) Level 2 Analog:**  A CNN is trained to predict a resistance class (e.g., Susceptible, Intermediate, Resistant) from a single segmented  *Klebsiella*  colony or inhibition zone.   
   **2\) Level 3 Analog:**  A Siamese CNN is trained to learn a similarity metric based on the  *visual phenotype of resistance* . For example, it would learn to map colonies with fuzzy edges near the inhibition zone close together in the embedding space, while mapping those with sharp, clear growth inhibition further apart. Clustering in this space would enforce a consistent resistance prediction for all colonies on a plate that exhibit a similar phenotypic response to the antibiotic, dramatically increasing the prediction's robustness.  
    
  **Problem Decomposition as a Design Pattern:**  The most profound insight from Signoroni et al. is the power of problem decomposition. The hierarchical architecture transforms an ambiguous, monolithic task ('interpret this plate') into a sequence of well-defined, verifiable, and more easily debugged sub-problems (Level 0: Segment all colonies \-\> Level 1: Select high-quality candidates \-\> Level 2: Classify individual candidates \-\> Level 3: Refine classifications using plate-level context \-\> Level 4: Apply clinical rules). This layered approach is a crucial design pattern for building robust and trustworthy AI systems in a clinical setting and should serve as the foundational blueprint for the thesis architecture.  
    
* **What gaps remain that my thesis could address?**  
  **Antibiotic Resistance Prediction:**  The most significant gap, and the central opportunity for the thesis, is that DeepColony performs  **species identification** , not  **antibiotic resistance prediction** . The thesis is perfectly positioned to address this by adapting the paper's powerful classification architecture to predict drug susceptibility instead of bacterial species.  
  **Inhibition Zone Analysis:**  The DeepColony workflow for species ID does not involve analyzing inhibition zones. The thesis must incorporate the segmentation and feature extraction of these zones, as their characteristics (size, clarity, shape) are primary visual indicators of antibiotic resistance.