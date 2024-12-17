# Setting Up a Virtual Environment and Installing Requirements

## Step 1: Install `venv`
Ensure you have Python installed. You can create a virtual environment using the `venv` module, which is included in Python 3.3 and later.

```bash
python3 -m venv myenv
```

This command creates a directory named myenv containing the virtual environment.

## Step 2: Activate the Virtual Environment
Activate the virtual environment using the following command:

On Windows:
```bash
    myenv\Scripts\activate
```
On macOS and Linux:
```bash
source myenv/bin/activate
```

## Step 3: Install Dependencies
With the virtual environment activated, install the dependencies listed in your requirements.txt file:
```
    pip install -r requirements.txt
```

## Step 4 Run:
Run The examples Provided in {ProjectRoot}/src/test_demo/ as well as 
{ProjectRoot}/src/image_template_matching/TemplateMatcher.py
The Yolo Implmentation in {ProjectRoot}/src/yolo_traind is still a work in progress

Unit test are stored in tests