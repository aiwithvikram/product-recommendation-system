from flask import Flask, render_template, request
import model
import os

app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['DEBUG'] = False

# Valid user IDs from the dataset
valid_userid = ['00sab00', '1234', 'zippy', 'zburt5', 'joshua', 'dorothy w', 'rebecca', 'walker557', 'samantha', 'raeanne', 'cimmie', 'cassie', 'moore222']

@app.route('/')
def view():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_top5():
    print(request.method)
    user_name = request.form['User Name']
    print('User name=', user_name)
    
    if user_name in valid_userid and request.method == 'POST':
        try:
            recommended_products = model.recommend_products(user_name)
            print('Recommended products:', recommended_products)
            
            if recommended_products and len(recommended_products) > 0:
                # Convert list to DataFrame for display
                import pandas as pd
                products_df = pd.DataFrame({'Product Name': recommended_products})
                
                return render_template('index.html', 
                                    column_names=products_df.columns.values, 
                                    row_data=list(products_df.values.tolist()), 
                                    zip=zip, 
                                    text='Recommended products')
            else:
                return render_template('index.html', text='No products found for this user')
                
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', text=f'Error processing recommendation: {str(e)}')
            
    elif user_name not in valid_userid:
        return render_template('index.html', text='No Recommendation found for the user')
    else:
        return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for load balancers"""
    return {'status': 'healthy', 'service': 'product-recommendation-system'}

if __name__ == '__main__':
    # For production, use environment variables
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    app.run(host=host, port=port)
