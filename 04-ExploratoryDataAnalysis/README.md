# Exploratory Data Analysis (EDA) Lecture Materials

**Course**: CMSC 173 - Data Science for Computer Scientists
**Topic**: Comprehensive EDA with Hands-on Implementation
**Author**: Course Materials

## 📁 Folder Contents

### 📊 Lecture Slides
- **`eda_lecture.tex`** - LaTeX Beamer source file (17 slides)
- **`eda_lecture.pdf`** - Compiled presentation slides
- **`eda_lecture.aux`**, **`eda_lecture.log`**, **`eda_lecture.nav`**, **`eda_lecture.out`** - LaTeX auxiliary files

### 💻 Interactive Companion
- **`eda_companion.ipynb`** - Google Colab notebook with complete code implementation

## 🎯 Learning Objectives

This comprehensive EDA module covers:

1. **Data Understanding & Types** (Slides 1-3)
   - Dataset overview and structure
   - Data type classification (numerical, categorical, binary, etc.)
   - Titanic dataset as practical example

2. **Univariate Analysis** (Slides 4-5)
   - Distribution analysis for numerical variables
   - Frequency analysis for categorical variables
   - Visualization techniques (histograms, box plots, bar charts)

3. **Bivariate Analysis** (Slide 6)
   - Correlation analysis
   - Cross-tabulations
   - Relationship exploration between variables

4. **Feature Selection** (Slide 7)
   - Statistical methods (F-test, Chi-square)
   - Tree-based importance
   - Sklearn implementation examples

5. **Missing Data Handling** (Slide 8)
   - Missing data patterns (MCAR, MAR, MNAR)
   - Imputation strategies
   - Visualization of missing data

6. **Feature Engineering** (Slide 9)
   - Creating new features
   - Feature transformation
   - Domain-specific insights

7. **Data Normalization** (Slide 10)
   - Min-Max scaling
   - Z-score standardization
   - Robust scaling techniques

8. **ML Pipeline Integration** (Slide 11)
   - End-to-end workflow
   - Model preparation
   - Performance evaluation

## 🚀 Quick Start

### For Instructors

1. **Compile LaTeX slides**:
   ```bash
   cd eda_lecture
   pdflatex eda_lecture.tex
   ```

2. **Open companion notebook**:
   - Upload `eda_companion.ipynb` to Google Colab
   - Or run locally with Jupyter Lab/Notebook

### For Students

1. **View slides**: Open `eda_lecture.pdf`
2. **Follow along**: Use `eda_companion.ipynb` for hands-on practice
3. **Practice**: Run all code cells and experiment with variations

## 📚 Slide Overview

| Slide | Topic | Key Concepts |
|-------|--------|-------------|
| 1 | Introduction | EDA definition, importance, workflow |
| 2 | Dataset Overview | Titanic dataset introduction |
| 3 | Data Types | Numerical, categorical, binary classification |
| 4 | Univariate - Numerical | Histograms, box plots, descriptive statistics |
| 5 | Univariate - Categorical | Bar charts, frequency tables |
| 6 | Bivariate Analysis | Correlation, cross-tabs, survival analysis |
| 7 | Feature Selection | Statistical tests, tree importance |
| 8 | Missing Data | Patterns, visualization, handling strategies |
| 9 | Feature Engineering | New feature creation, transformation |
| 10 | Normalization | Scaling techniques comparison |
| 11 | ML Pipeline | Complete workflow integration |
| 12-17 | Advanced Topics | Correlation analysis, transformations, etc. |

## 💡 Notebook Features

The companion Jupyter notebook includes:

### 📊 **Interactive Visualizations**
- Matplotlib and Seaborn plots
- Plotly for interactive charts
- Missing data heatmaps
- Correlation matrices

### 🔧 **Hands-on Implementation**
- Complete EDA workflow
- Feature engineering examples
- Data preprocessing pipeline
- Model training and evaluation

### 📝 **Educational Content**
- Detailed markdown explanations
- Code comments and documentation
- Best practices and tips
- Real-world insights

### 🎯 **Practical Examples**
- Titanic dataset analysis
- Survival prediction modeling
- Feature importance analysis
- Performance metrics evaluation

## 🛠️ Technical Requirements

### LaTeX Compilation
- **Engine**: pdfLaTeX
- **Theme**: Metropolis with Wolverine color scheme
- **Packages**: TikZ, pgfplots, booktabs, array
- **Font Size**: 8pt
- **Aspect Ratio**: 16:10

### Notebook Dependencies
```python
# Core libraries
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0

# Machine learning
scikit-learn >= 1.0.0
scipy >= 1.7.0

# Interactive visualization
plotly >= 5.0.0

# Optional for enhanced experience
jupyter >= 1.0.0
ipywidgets >= 7.6.0
```

## 🎨 Customization

### Modifying Slides
- Edit `eda_lecture.tex` for content changes
- Maintain Metropolis theme consistency
- Use TikZ for custom diagrams
- Follow 8pt font size for readability

### Adapting Notebook
- Replace Titanic dataset with your own data
- Modify feature engineering based on domain
- Adjust visualization styles
- Add domain-specific analysis

## 📖 Usage Scenarios

### 🎓 **Academic Course**
- Use slides for 50-minute lecture
- Assign notebook as homework/lab
- Combine with theory discussions
- Build upon for advanced topics

### 🏢 **Professional Training**
- Workshop format (2-3 hours)
- Interactive coding sessions
- Real-world case studies
- Team-based analysis projects

### 📚 **Self-Study**
- Work through slides and notebook sequentially
- Practice with different datasets
- Experiment with code modifications
- Build portfolio projects

## 🔗 Related Resources

- **Course Repository**: [CMSC173 Main](../../)
- **Other Lectures**: [Linear Regression](../01-LinearRegression/), [Logistic Regression](../02-LogisticRegression/)
- **External Links**:
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [Seaborn Gallery](https://seaborn.pydata.org/examples/)
  - [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)

## 🤝 Contributing

To improve these materials:

1. **Content Updates**: Edit LaTeX or notebook files
2. **Bug Fixes**: Report issues with code or slides
3. **Enhancements**: Add new visualizations or examples
4. **Documentation**: Improve explanations and comments

## 📄 License

Educational materials for CMSC 173. Please cite appropriately when using in other courses or materials.

---

**🎉 Happy Learning and Teaching!**

*For questions or support, please contact the course instructor.*