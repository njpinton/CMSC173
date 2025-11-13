from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <h1>Presentations</h1>
    <ul>
        <li><a href="/module/0">Module 0: Introduction to Machine Learning</a></li>
        <li><a href="/module/1">Module 1: Parameter Estimation</a></li>
        <li><a href="/module/2">Module 2: Linear Regression</a></li>
        <li><a href="/module/4">Module 4: Exploratory Data Analysis</a></li>
    </ul>
    """

@app.route('/module/<int:module_number>')
def show_module(module_number):
    if module_number == 0:
        return render_template('module_00.html')
    elif module_number == 1:
        return render_template('module_01.html')
    elif module_number == 2:
        return render_template('module_02.html')
    elif module_number == 4:
        return render_template('module_04.html')
    else:
        return "Module not found", 404

if __name__ == '__main__':
    app.run(debug=True, port=8788)
