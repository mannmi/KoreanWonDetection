# Korean Won Detection Repository

## Repository Setup

First, clone the repository using the following command:

```bash
git clone https://github.com/mannmi/KoreanWonDetection.git
```

---

## Setting Up a Virtual Environment and Installing Requirements

### Step 1: Install `venv`
Ensure you have Python installed. You can create a virtual environment using the `venv` module, which is included in Python 3.3 and later.

```bash
python3 -m venv myenv
```

This command creates a directory named myenv containing the virtual environment.

### Step 2: Activate the Virtual Environment
Activate the virtual environment using the following command:

On Windows:
```bash
    myenv\Scripts\activate
```
On macOS and Linux:
```bash
source myenv/bin/activate
```

### Step 3: Install Dependencies
With the virtual environment activated, install the dependencies listed in your requirements.txt file:
```
    pip install -r requirements.txt
```

## Step 4 Run:
Run The examples Provided in {ProjectRoot}/src/test_demo/ as well as 
{ProjectRoot}/src/image_template_matching/TemplateMatcher.py
The Yolo Implmentation in {ProjectRoot}/src/yolo_traind is still a work in progress

Unit test are stored in tests

---

## Recommended Algorithms

Within the `src` folder, we recommend focusing on two primary algorithm directories:

### 1. **Image Template Matching**  
- **Location**: `src/image_template_matching`  
- **Purpose**: Detecting and recognizing Korean Won note features using template matching techniques.  

### 2. **Edge Detection**  
- **Location**: `src/edge_detection`  
- **Purpose**: Identifying boundaries and key features of currency images.  

---

## Getting Started

To begin using the repository:  
1. Explore the recommended algorithm directories.  
2. Experiment with different detection methods.  
3. Understand the implementation details.  
4. Test the algorithms with various Korean Won note images.  
=======
# Korean Won Detection Repository

## Repository Setup

First, clone the repository using the following command:

```bash
git clone https://github.com/mannmi/KoreanWonDetection.git
```

---

## Recommended Algorithms

Within the `src` folder, we recommend focusing on two primary algorithm directories:

### 1. **Image Template Matching**  
- **Location**: `src/image_template_matching`  
- **Purpose**: Detecting and recognizing Korean Won note features using template matching techniques.  

### 2. **Edge Detection**  
- **Location**: `src/edge_detection`  
- **Purpose**: Identifying boundaries and key features of currency images.  

---

## Getting Started

To begin using the repository:  
1. Explore the recommended algorithm directories.  
2. Experiment with different detection methods.  
3. Understand the implementation details.  
4. Test the algorithms with various Korean Won note images.  

---

## Dependencies

Ensure all necessary dependencies are installed before running the algorithms. Refer to the repository's `README` for specific setup instructions.
