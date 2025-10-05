from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

# Load data once when app starts
df = pd.read_csv(r'C:\Users\ashwi\Music\txt ai agent\worldometer_data.csv')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Groq API configuration - Replace with your actual API key
GROQ_API_KEY = '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'  # Replace this with your actual key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def ask_llm(prompt):
    """Call Groq API with DeepSeek-R1 model"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a code generator. Only output executable pandas code, no explanations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_pandas_code(question):
    prompt = f"""
You are a pandas expert analyzing COVID-19 data. Dataset columns:
{list(df.columns)}

Convert this question to ONE LINE of pandas code using 'df':
"{question}"

CRITICAL RULES:
- Return ONLY the pandas expression, nothing else
- No explanations, no thinking process, no comments
- Just the executable code: df.something()
- Use .idxmax() to get the index name, not .max()
- Column names with special characters: df['Country/Region']

Examples:
"country with most cases" → df.loc[df['TotalCases'].idxmax(), 'Country/Region']
"total cases by continent" → df.groupby('Continent')['TotalCases'].sum().sort_values(ascending=False)
"top 10 countries by deaths" → df.nlargest(10, 'TotalDeaths')[['Country/Region', 'TotalDeaths']]

Code only:
"""
    
    response = ask_llm(prompt)
    
    # Remove thinking tags if present (DeepSeek-R1 specific)
    if '<think>' in response:
        response = response.split('</think>')[-1].strip()
    
    # Clean code blocks
    if '```' in response:
        # Extract code between ``` markers
        parts = response.split('```')
        for part in parts:
            clean_part = part.replace('python', '').strip()
            if 'df' in clean_part and len(clean_part) < 200:
                response = clean_part
                break
    
    # Get the actual pandas line
    lines = response.split('\n')
    
    # First try: lines starting with df
    for line in lines:
        line = line.strip()
        if line.startswith('df.') or line.startswith('df['):
            # Remove any trailing comments
            if '#' in line:
                line = line.split('#')[0].strip()
            return line
    
    # Second try: lines containing df with assignment
    for line in lines:
        line = line.strip()
        if ('df.' in line or 'df[' in line) and '=' in line:
            # Extract right side of assignment
            code = line.split('=', 1)[-1].strip()
            if '#' in code:
                code = code.split('#')[0].strip()
            return code
    
    # Third try: any line with df
    for line in lines:
        if 'df' in line and not line.startswith('#'):
            line = line.strip()
            if '#' in line:
                line = line.split('#')[0].strip()
            return line
    
    # Fallback
    return response.strip()

def generate_explanation(question, result_data):
    """Generate natural language explanation of the results"""
    
    # Prepare result summary
    if isinstance(result_data, pd.DataFrame):
        summary = f"DataFrame with {len(result_data)} rows. Top values:\n{result_data.head(10).to_string()}"
    elif isinstance(result_data, pd.Series):
        summary = f"Series with {len(result_data)} values:\n{result_data.head(10).to_string()}"
    else:
        summary = str(result_data)
    
    prompt = f"""
You are a public health analyst. A user asked: "{question}"

Data results:
{summary}

Provide a 2-3 sentence explanation in plain English. Focus on key insights and numbers.
DO NOT write code. DO NOT use markdown. Just write normal sentences explaining the data.

Explanation:
"""
    
    explanation = ask_llm(prompt)
    
    # Clean up any code blocks that might appear
    if '```' in explanation:
        explanation = explanation.split('```')[0].strip()
    
    # Remove any remaining code-like content
    lines = []
    for line in explanation.split('\n'):
        if not line.strip().startswith('import') and not line.strip().startswith('df') and 'pd.DataFrame' not in line:
            lines.append(line)
    
    return '\n'.join(lines).strip()

def should_show_visualization(question):
    """Determine if user wants to see a chart"""
    chart_keywords = ['graph', 'chart', 'plot', 'visualize', 'show chart', 'show graph', 'bar chart', 'pie chart', 'compare visually']
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in chart_keywords)

def should_show_table(question):
    """Determine if user wants to see detailed table"""
    table_keywords = ['show all', 'list all', 'table', 'full list', 'complete list', 'show details']
    question_lower = question.lower()
    
    # Also show table for "top X" or "bottom X" queries
    if 'top ' in question_lower or 'bottom ' in question_lower:
        return True
    
    return any(keyword in question_lower for keyword in table_keywords)
    """Automatically create appropriate visualization"""
    
    # Single value - no chart needed
    if isinstance(result_data, (int, float, str)):
        return None
    
    # Convert to DataFrame if Series
    if isinstance(result_data, pd.Series):
        result_df = result_data.reset_index()
        result_df.columns = ['Category', 'Value']
    else:
        result_df = result_data
    
    # Too many rows - show table only
    if len(result_df) > 20:
        return None
    
    # Determine chart type based on data
    try:
        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = result_df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            # Bar chart for categorical vs numeric
            fig = px.bar(
                result_df, 
                x=categorical_cols[0], 
                y=numeric_cols[0],
                title=f"{question}",
                template="plotly_white"
            )
            fig.update_layout(height=400)
            return fig.to_json()
        elif len(numeric_cols) >= 2:
            # Scatter or line chart
            fig = px.scatter(
                result_df,
                x=result_df.columns[0],
                y=result_df.columns[1],
                title=f"{question}",
                template="plotly_white"
            )
            fig.update_layout(height=400)
            return fig.to_json()
    except:
        pass
    
    return None

def format_table(result_data):
    """Format result as HTML table"""
    
    if isinstance(result_data, (int, float)):
        return f"<div class='single-value'>{result_data:,.0f}</div>"
    
    if isinstance(result_data, str):
        return f"<div class='single-value'>{result_data}</div>"
    
    if isinstance(result_data, pd.Series):
        result_df = result_data.reset_index()
        result_df.columns = ['Category', 'Value']
    else:
        result_df = result_data
    
    # Limit rows for display
    display_df = result_df.head(50)
    
    # Format numbers
    for col in display_df.select_dtypes(include=['number']).columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
    
    html = display_df.to_html(index=False, classes='table table-striped table-hover')
    
    if len(result_df) > 50:
        html += f"<p class='text-muted'><small>Showing first 50 of {len(result_df)} rows</small></p>"
    
    return html

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided', 'success': False})
    
    try:
        # Generate pandas code
        code = generate_pandas_code(question)
        print(f"Generated code: {code}")  # Debug print
        
        # Execute code
        result = eval(code)
        
        # Check if result is empty
        if result is None or (hasattr(result, '__len__') and len(result) == 0):
            return jsonify({
                'success': False,
                'question': question,
                'error': 'No data found for this query. Please try rephrasing your question.'
            })
        
        # Generate table HTML only if user wants it
        show_table = should_show_table(question)
        if show_table:
            table_html = format_table(result)
        else:
            # Show simple answer for single values
            if isinstance(result, (int, float, str)):
                table_html = f"<div class='single-value'>{result}</div>"
            else:
                table_html = None
        
        # Generate visualization only if user asks
        show_chart = should_show_visualization(question)
        if show_chart:
            chart_json = create_visualization(result, question)
        else:
            chart_json = None
        
        # Generate explanation
        explanation = generate_explanation(question, result)
        
        return jsonify({
            'success': True,
            'question': question,
            'table': table_html,
            'chart': chart_json,
            'explanation': explanation
        })
        
    except Exception as e:
        print(f"Error executing code: {code if 'code' in locals() else 'N/A'}")  # Debug print
        return jsonify({
            'success': False,
            'question': question,
            'error': f'{str(e)} (Generated code: {code if "code" in locals() else "unknown"})'
        })

if __name__ == '__main__':
    app.run(debug=True)