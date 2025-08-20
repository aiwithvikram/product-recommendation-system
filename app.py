from flask import Flask, render_template, request
import model_optimized as model

app = Flask(__name__)

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
            top20_products = model.recommend_products(user_name)
            print(top20_products.head())
            
            if not top20_products.empty:
                get_top5 = model.top5_products(top20_products)
                if not get_top5.empty:
                    return render_template('index.html', 
                                        column_names=get_top5.columns.values, 
                                        row_data=list(get_top5.values.tolist()), 
                                        zip=zip, 
                                        text='Recommended products')
                else:
                    return render_template('index.html', text='No positive sentiment products found for recommendations')
            else:
                return render_template('index.html', text='No products found for this user')
                
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', text=f'Error processing recommendation: {str(e)}')
            
    elif user_name not in valid_userid:
        return render_template('index.html', text='No Recommendation found for the user')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.debug = False
    # For local development
    app.run(host='0.0.0.0', port=5000)
