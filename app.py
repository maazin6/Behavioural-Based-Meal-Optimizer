from flask import Flask, render_template, request, redirect, flash, jsonify, session, url_for
import psycopg2
from psycopg2 import sql
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import re

import pandas as pd
app = Flask(__name__)
app.secret_key = 'supersecretkey'


# Database configuration
DB_HOST = "localhost"
DB_NAME = "behaviour"
DB_USER = "postgres"
DB_PASSWORD = "2002"

# Helper function to connect to the database
def connect_to_db():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# Route to render the signup page
@app.route('/signup')
def signup():
    return render_template('signup.html')

# Route to check the signup form
@app.route('/check_signup', methods=['POST'])
def check_signup():
    username = request.form.get('username')
    password = request.form.get('password')
    retype_password = request.form.get('retype_password')

    errors = []
    if not username:
        errors.append("Username is required.")
    if password != retype_password:
        errors.append("Passwords do not match.")

    # Check if username already exists
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user_exists = cursor.fetchone() is not None
                if user_exists:
                    errors.append("Username already exists.")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        errors.append("An error occurred. Please try again.")

    if errors:
        return jsonify({"success": False, "messages": errors}), 400
    else:
        return jsonify({"success": True, "messages": ["Validation successful!"]}), 200

# Route to handle form submission from the signup page
@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    username = request.form.get('username')
    password = request.form.get('password')
    age = request.form.get('age')
    gender = request.form.get('gender')
    region = request.form.get('region')
    meal_type = request.form.get('meal_type')

    if not (username and password and age and gender and region and meal_type):
        flash("All fields are required.")
        return render_template('signup.html')

    allergies = ','.join(request.form.getlist('allergies'))
    #hashed_password = generate_password_hash(password)

    preferences = {
        'paneer': int(request.form.get('paneer', 0)),
        'soy': int(request.form.get('soy', 0)),
        'legumes_and_lentils': int(request.form.get('legumes_and_lentils', 0)),
        'leafy_greens': int(request.form.get('leafy_greens', 0)),
        'root_vegetables': int(request.form.get('root_vegetables', 0)),
        'mushrooms': int(request.form.get('mushrooms', 0)),
        'mutton': int(request.form.get('mutton', 0)),
        'chicken': int(request.form.get('chicken', 0)),
        'beef': int(request.form.get('beef', 0)),
        'fish': int(request.form.get('fish', 0)),
        'prawn': int(request.form.get('prawn', 0)),
        'eggs': int(request.form.get('eggs', 0)),
        'allergies': allergies
    }

    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO users (username, password, age, gender, region, meal_type, paneer, soy, legumes_and_lentils,
                                       leafy_greens, root_vegetables, mushrooms, mutton, chicken, beef, fish, prawn, eggs, allergies)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (username, password, age, gender, region, meal_type,
                      preferences['paneer'], preferences['soy'], preferences['legumes_and_lentils'],
                      preferences['leafy_greens'], preferences['root_vegetables'], preferences['mushrooms'],
                      preferences['mutton'], preferences['chicken'], preferences['beef'],
                      preferences['fish'], preferences['prawn'], preferences['eggs'], preferences['allergies']))
                
                conn.commit()
                flash("Signup successful! Please log in.")
                return redirect('/login')

    except psycopg2.Error as e:
        flash("An error occurred. Please try again.")
        print(f"Database error: {e}")
        return redirect('/signup')

def get_user_by_username(username):
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if row:
                return {
                    'user_id': row[0],
                    'username': row[1],
                    'password': row[2]  # Ensure the password is hashed if you're using hashing
                }
    return None

# Route to render the login page
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    data = request.form
    username = data.get('username')
    password = data.get('password')

    # Retrieve user by username
    user = get_user_by_username(username)
    
    # Validate credentials
    if user and user['password'] == password:  # Direct comparison for plain-text passwords
        session['user_id'] = user['user_id']
        return redirect('/mood')
    else:
        flash("Invalid username or password")
        return redirect('/login')


# Route for the main page (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.")
    return redirect('/login')



@app.route('/mood')
def mood():
    return render_template('mood.html')


breakfast_model = joblib.load('breakfast_model.pkl')
lunch_model = joblib.load('lunch_model.pkl')
dinner_model = joblib.load('dinner_model.pkl')

# Load label encoders for breakfast, lunch, and dinner
breakfast_label_encoders = joblib.load('breakfast_label_encoders.pkl')
lunch_label_encoders = joblib.load('lunch_label_encoders.pkl')
dinner_label_encoders = joblib.load('dinner_label_encoders.pkl')

# Assume target encoder is the same for all models
target_encoder = breakfast_label_encoders['meal_name']

def get_user_details(user_id):
    """Fetch user details from the database if not in session."""
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT age, gender, region, meal_type FROM users WHERE id = %s
            """, (user_id,))
            user = cursor.fetchone()
            if user:
                return {
                    'age': user[0],
                    'gender': user[1],
                    'region': user[2],
                    'meal_type': user[3]
                }
    return {}

# Load models and encoders once
breakfast_model = joblib.load('breakfast_model.pkl')
lunch_model = joblib.load('lunch_model.pkl')
dinner_model = joblib.load('dinner_model.pkl')

breakfast_label_encoders = joblib.load('breakfast_label_encoders.pkl')
lunch_label_encoders = joblib.load('lunch_label_encoders.pkl')
dinner_label_encoders = joblib.load('dinner_label_encoders.pkl')

# Define target encoders specific to each meal type
breakfast_target_encoder = breakfast_label_encoders['meal_name']
lunch_target_encoder = lunch_label_encoders['meal_name']
dinner_target_encoder = dinner_label_encoders['meal_name']

def predict_meal_name(model, label_encoders, target_encoder, age, gender, region, meal_type, mood, stress_level, allergies):
    # Prepare the input data as a dictionary
    user_input = {
        'age': age,
        'gender': gender,
        'region': region,
        'meal_type': meal_type,
        'mood': mood,
        'stress_level': stress_level,
        'allergies': allergies
    }
    
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical features using the provided label encoders
    for col in input_df.columns:
        if col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                print(f"Unseen label '{input_df[col].values[0]}' encountered in column '{col}'. Assigning default value.")
                default_value = label_encoders[col].classes_[0]
                input_df[col] = label_encoders[col].transform([default_value])

    # Make a prediction
    prediction = model.predict(input_df)

    # Decode the predicted meal name
    meal_name = target_encoder.inverse_transform(prediction)[0]

    return meal_name

def split_ingredients(ingredients_str):
    # Split on commas that are not inside parentheses
    return re.split(r',\s*(?![^(]*\))', ingredients_str)

@app.route('/submit_mood', methods=['POST'])
def submit_mood():
    # Get mood and stress level from the form
    mood = request.form.get('mood')
    stress_level = int(request.form.get('stress_level', 0))

    # Get user details from session or database
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to continue.")
        return redirect(url_for('login'))

    # Retrieve user details from session or database
    user_details = {
        'age': session.get('age'),
        'gender': session.get('gender'),
        'region': session.get('region'),
        'meal_type': session.get('meal_type', 'all'),
        'allergies': session.get('allergies', 'none')
    }
    if None in user_details.values():  # If any detail is missing, fetch from the DB
        user_details.update(get_user_details(user_id))

    # Predict for breakfast using the breakfast target encoder
    recommended_breakfast = predict_meal_name(
        model=breakfast_model,
        label_encoders=breakfast_label_encoders,
        target_encoder=breakfast_target_encoder,
        age=user_details['age'],
        gender=user_details['gender'],
        region=user_details['region'],
        meal_type=user_details['meal_type'],
        mood=mood,
        stress_level=stress_level,
        allergies=user_details['allergies']
    )

    # Predict for lunch using the lunch target encoder
    recommended_lunch = predict_meal_name(
        model=lunch_model,
        label_encoders=lunch_label_encoders,
        target_encoder=lunch_target_encoder,
        age=user_details['age'],
        gender=user_details['gender'],
        region=user_details['region'],
        meal_type=user_details['meal_type'],
        mood=mood,
        stress_level=stress_level,
        allergies=user_details['allergies']
    )

    # Predict for dinner using the dinner target encoder
    recommended_dinner = predict_meal_name(
        model=dinner_model,
        label_encoders=dinner_label_encoders,
        target_encoder=dinner_target_encoder,
        age=user_details['age'],
        gender=user_details['gender'],
        region=user_details['region'],
        meal_type=user_details['meal_type'],
        mood=mood,
        stress_level=stress_level,
        allergies=user_details['allergies']
    )

    # Print the meal recommendations in the console
    print(f"Recommended Breakfast for User {user_id}: {recommended_breakfast}")
    print(f"Recommended Lunch for User {user_id}: {recommended_lunch}")
    print(f"Recommended Dinner for User {user_id}: {recommended_dinner}")

    # Insert recommended meals into the database
    insert_menu_data(
        id=user_id,
        mood=mood,
        stress_level=stress_level,
        breakfast=recommended_breakfast,
        lunch=recommended_lunch,
        dinner=recommended_dinner
    )

    # Flash message and redirect as needed
    #flash("Meal plans for breakfast, lunch, and dinner have been generated.")
    return redirect('/menu')


def insert_menu_data(id, mood, stress_level, breakfast, lunch, dinner):
    # Database connection parameters
    db_params = {
        'dbname': 'behaviour',
        'user': 'postgres',
        'password': '2002',
        'host': 'localhost',  # or your database host
        
    }

    # SQL insert query
    insert_query = sql.SQL("""
        INSERT INTO menu_data (id, mood, stress_level, breakfast, lunch, dinner)
        VALUES (%s, %s, %s, %s, %s, %s)
    """)

    try:
        # Establish connection
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # Execute the query with data
        cur.execute(insert_query, (id, mood, stress_level, breakfast, lunch, dinner))

        # Commit the transaction
        conn.commit()

        print("Data inserted successfully into menu_data table")

    except psycopg2.DatabaseError as error:
        print("Error:", error)

    finally:
        # Close the cursor and connection
        if cur:
            cur.close()
        if conn:
            conn.close()



# Database connection parameters
db_params = {
    'dbname': 'behaviour',
    'user': 'postgres',
    'password': '2002',
    'host': 'localhost',  # or your database host
}

# Function to retrieve meal names and images for display
def get_menu_with_images(user_id):
    try:
        # Establish a single database connection
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # Query to fetch the latest menu for the user along with meal images and IDs
        cur.execute("""
            SELECT md.breakfast, b.image AS breakfast_image, b.meal_id AS breakfast_id,
                   md.lunch, l.image AS lunch_image, l.meal_id AS lunch_id,
                   md.dinner, d.image AS dinner_image, d.meal_id AS dinner_id
            FROM menu_data md
            LEFT JOIN meals b ON md.breakfast = b.meal_name
            LEFT JOIN meals l ON md.lunch = l.meal_name
            LEFT JOIN meals d ON md.dinner = d.meal_name
            WHERE md.id = %s
            ORDER BY md.created_at DESC LIMIT 1
        """, (user_id,))

        # Fetch the result
        result = cur.fetchone()
        if result:
            # Unpack results including meal IDs
            (breakfast_name, breakfast_image, breakfast_id,
             lunch_name, lunch_image, lunch_id,
             dinner_name, dinner_image, dinner_id) = result

            # If an image is missing, set it to None to prevent display in the template
            breakfast_image = breakfast_image if breakfast_image else None
            lunch_image = lunch_image if lunch_image else None
            dinner_image = dinner_image if dinner_image else None

            # Return data in a dictionary format suitable for templating
            return {
                'breakfast_name': breakfast_name,
                'breakfast_image': breakfast_image,
                'breakfast_id': breakfast_id,
                'lunch_name': lunch_name,
                'lunch_image': lunch_image,
                'lunch_id': lunch_id,
                'dinner_name': dinner_name,
                'dinner_image': dinner_image,
                'dinner_id': dinner_id
            }
        else:
            print("No menu data found for user.")
            return None

    except psycopg2.DatabaseError as error:
        print("Error fetching menu with images:", error)
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


@app.route('/menu')
def menu():
    # Check if user is logged in
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to view the menu.")
        return redirect(url_for('login'))
    
    # Fetch the latest menu data for the user
    menu_data = get_menu_with_images(user_id)

    if menu_data:
        # Debug print to verify the data being passed to the template
        print("Menu data:", menu_data)

        # Render the menu template with meal names and image URLs
        return render_template('menu.html', **menu_data)
    else:
        flash("No menu data available for your account.")
        return redirect(url_for('login'))

@app.route('/meal_details/<int:meal_id>')
def meal_details(meal_id):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Fetch meal details by meal_id
    cursor.execute('SELECT meal_name, image, ingredients, recipe, calories, proteins, fats, carbohydrates, fibre FROM meals WHERE meal_id = %s', (meal_id,))
    meal = cursor.fetchone()

    cursor.close()
    conn.close()

    # Convert ingredients with the custom split function and instructions by newlines
    if meal:
        meal_data = {
            'name': meal[0],
            'image': meal[1],
            'ingredients': split_ingredients(meal[2]) if meal[2] else [],
            'instructions': meal[3].split('$'),
            'nutrition': {
                'calories': meal[4],
                'protein': meal[5],
                'fat': meal[6],
                'carbohydrate': meal[7],
                'fiber': meal[8]
            }
        }
    else:
        meal_data = None

    # Render the template with structured meal data
    return render_template('meal_details.html', meal=meal_data)

def get_available_meals(user_id, meal_type):
    conn = connect_to_db()
    cur = conn.cursor()

    # Fetch user's meal type preference and allergies
    cur.execute("SELECT meal_type, allergies FROM users WHERE id = %s", (user_id,))
    user_preferences = cur.fetchone()
    if not user_preferences:
        print("User preferences not found.")
        return []

    user_meal_type, user_allergies = user_preferences

    # Prepare allergy filter
    allergy_list = user_allergies.split(',') if user_allergies else []
    allergy_conditions = " AND ".join([f"meals.allergies NOT LIKE '%{allergy.strip()}%'" for allergy in allergy_list])

    # Form the SQL query, dynamically adjusting for allergies
    sql_query = f"""
        SELECT meal_id, meal_name, image
        FROM meals
        WHERE meal_type = %s
          AND (meals.is_vegetarian = TRUE OR meals.is_vegetarian = %s)
    """
    # Add allergy conditions if present
    if allergy_conditions:
        sql_query += f" AND ({allergy_conditions})"

    print("Executing Query:", sql_query)
    print("With Parameters:", meal_type, user_meal_type == "vegetarian")

    # Execute the query with only the required parameters
    cur.execute(sql_query, (meal_type, user_meal_type == "vegetarian"))

    available_meals = cur.fetchall()
    cur.close()
    conn.close()
    
    return available_meals

def get_filtered_meals(user_id, meal_time):
    """
    Fetch meals based on the user's meal_type preference (vegetarian or non-vegetarian)
    and the specified meal_time (breakfast, lunch, or dinner).
    
    Parameters:
        user_id (int): The ID of the user.
        meal_time (str): The meal time to filter by (e.g., 'breakfast', 'lunch', 'dinner').

    Returns:
        list: A list of meals that match the criteria, each including meal_id, name, and image.
    """
    conn = connect_to_db()
    cur = conn.cursor()

    # Fetch user's meal type preference (vegetarian or non-vegetarian)
    cur.execute("SELECT meal_type FROM users WHERE id = %s", (user_id,))
    user_meal_type = cur.fetchone()[0]  # Assume this returns either 'vegetarian' or 'non-vegetarian'

    # Form the SQL query based on user's meal type preference and meal time
    sql_query = """
        SELECT meal_id, meal_name, image
        FROM meals
        WHERE meal_time = %s
          AND meal_type = %s
    """
    params = [meal_time, user_meal_type]  # Parameters include meal_time and user_meal_type

    print("Executing Query:", sql_query)
    print("With Parameters:", params)

    # Execute the query
    cur.execute(sql_query, params)

    available_meals = cur.fetchall()
    cur.close()
    conn.close()
    
    return available_meals


def perform_swap_meal(user_id, meal_type, new_meal_id):
    """
    Perform a meal swap for a specific meal type in the user's current menu.
    
    Parameters:
        user_id (int): The ID of the user.
        meal_type (str): The type of meal to swap (e.g., 'breakfast', 'lunch', 'dinner').
        new_meal_id (int): The ID of the new meal to replace the current meal.

    Returns:
        bool: True if the swap was successful, False otherwise.
    """
    conn = connect_to_db()
    cur = conn.cursor()

    try:
        # Get the meal_name associated with the new meal_id
        cur.execute("SELECT meal_name FROM meals WHERE meal_id = %s", (new_meal_id,))
        result = cur.fetchone()
        
        if not result:
            print("Meal with the specified ID not found.")
            return False

        new_meal_name = result[0]

        # Get the latest menu_data id for the user
        cur.execute("""
            SELECT id, {meal_type} FROM menu_data
            WHERE id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """.format(meal_type=meal_type), (user_id,))
        
        latest_menu_entry = cur.fetchone()
        
        if latest_menu_entry:
            latest_menu_id = latest_menu_entry[0]
            current_meal = latest_menu_entry[1]

            # Only update if the meal name is different to avoid redundant updates
            if current_meal == new_meal_name:
                print(f"No update needed; {meal_type} is already set to '{new_meal_name}'.")
                return False

            # Update the specific meal type column (breakfast, lunch, or dinner) in the latest menu entry with the new meal name
            cur.execute(f"""
                UPDATE menu_data
                SET {meal_type} = %s
                WHERE id = %s
            """, (new_meal_name, latest_menu_id))

            conn.commit()
            print(f"Successfully swapped {meal_type} for user {user_id} to meal '{new_meal_name}'")
            return True
        else:
            print("No menu data found for the specified user.")
            return False

    except Exception as e:
        print("Error performing meal swap:", e)
        conn.rollback()
        return False

    finally:
        cur.close()
        conn.close()



@app.route('/swap/<meal_type>', methods=['GET', 'POST'])
def swap_meal(meal_type):
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to swap meals.")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Retrieve selected meal ID from form
        new_meal_id = request.form.get('meal_id')
        success = perform_swap_meal(user_id, meal_type, new_meal_id)
        
        if success:
            flash(f"{meal_type.capitalize()} meal has been successfully swapped.")
        else:
            flash(f"Failed to swap {meal_type} meal. Please try again.")
        return redirect(url_for('menu'))

    # Get available meals filtered by user preferences and meal type
    available_meals = get_filtered_meals(user_id, meal_type)
    return render_template('swap_meal.html', meals=available_meals, meal_type=meal_type)

@app.route('/mealdatabase')
def mealdatabase():
    conn = connect_to_db()
    cur = conn.cursor()

    try:
        # Fetch all meals from the meals table
        cur.execute("SELECT meal_name, image, meal_id FROM meals")
        meals = cur.fetchall()

    except Exception as e:
        print("Error fetching meals:", e)
        meals = []
    finally:
        cur.close()
        conn.close()

    # Render the template with the list of meals
    return render_template('mealdatabase.html', meals=meals)

@app.route('/profile', methods=['GET'])
def profile():
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to view or edit your profile.")
        return redirect(url_for('login'))

    conn = connect_to_db()
    cur = conn.cursor()

    # Fetch the user's profile data
    cur.execute("SELECT age, gender, region, meal_type, password, allergies FROM users WHERE id = %s", (user_id,))
    profile_data = cur.fetchone()

    if profile_data:
        age, gender, region, meal_type, password, allergies = profile_data
        allergies = allergies.split(",") if allergies else []  # Convert allergies string to a list
    else:
        flash("Profile data not found.")
        return redirect(url_for('menu'))

    cur.close()
    conn.close()

    # Render profile with user data
    return render_template('profile.html', age=age, gender=gender, region=region, meal_type=meal_type, password=password, allergies=allergies)


@app.route('/update_password', methods=['POST'])
def update_password():
    user_id = session.get('user_id')
    new_password = request.form.get('password')
    if user_id and new_password:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET password = %s WHERE id = %s", (new_password, user_id))
        conn.commit()
        cur.close()
        conn.close()
        flash("Password updated successfully.", "success")
    return redirect(url_for('profile'))


@app.route('/update_age', methods=['POST'])
def update_age():
    user_id = session.get('user_id')
    new_age = request.form.get('age')
    if user_id and new_age:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET age = %s WHERE id = %s", (new_age, user_id))
        conn.commit()
        cur.close()
        conn.close()
        flash("Age group updated successfully.", "success")
    return redirect(url_for('profile'))


@app.route('/update_gender', methods=['POST'])
def update_gender():
    user_id = session.get('user_id')
    new_gender = request.form.get('gender')
    if user_id and new_gender:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET gender = %s WHERE id = %s", (new_gender, user_id))
        conn.commit()
        cur.close()
        conn.close()
        flash("Gender updated successfully.", "success")
    return redirect(url_for('profile'))


@app.route('/update_region', methods=['POST'])
def update_region():
    user_id = session.get('user_id')
    new_region = request.form.get('region')
    if user_id and new_region:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET region = %s WHERE id = %s", (new_region, user_id))
        conn.commit()
        cur.close()
        conn.close()
        flash("Region updated successfully.", "success")
    return redirect(url_for('profile'))


@app.route('/update_meal_type', methods=['POST'])
def update_meal_type():
    user_id = session.get('user_id')
    new_meal_type = request.form.get('meal_type')
    if user_id and new_meal_type:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET meal_type = %s WHERE id = %s", (new_meal_type, user_id))
        conn.commit()
        cur.close()
        conn.close()
        flash("Meal type updated successfully.", "success")
    return redirect(url_for('profile'))


@app.route('/update_allergies', methods=['POST'])
def update_allergies():
    user_id = session.get('user_id')
    new_allergies = request.form.getlist('allergies')
    if user_id and new_allergies:
        new_allergies_str = ",".join(new_allergies)  # Convert list to comma-separated string
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET allergies = %s WHERE id = %s", (new_allergies_str, user_id))
        conn.commit()
        cur.close()
        conn.close()
        flash("Allergies updated successfully.", "success")
    return redirect(url_for('profile'))


if __name__ == '__main__':
    app.run(debug=True)
