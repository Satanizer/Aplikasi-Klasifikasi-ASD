from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
import joblib
import numpy as np
from pytz import timezone
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = '12345'

# Konfigurasi database menggunakan MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://flaskuser:password_anda@localhost/predictions_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Model Database: User untuk akun
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Fungsi untuk mendapatkan waktu sekarang di zona waktu Asia/Jakarta
def jakarta_time():
    return datetime.datetime.now(timezone('Asia/Jakarta'))

class Prediction(db.Model):
    id = db.Column(db.Integer, nullable=False)
    A1 = db.Column(db.Float, nullable=False)
    A2 = db.Column(db.Float, nullable=False)
    A3 = db.Column(db.Float, nullable=False)
    A4 = db.Column(db.Float, nullable=False)
    A5 = db.Column(db.Float, nullable=False)
    A6 = db.Column(db.Float, nullable=False)
    A7 = db.Column(db.Float, nullable=False)
    A8 = db.Column(db.Float, nullable=False)
    A9 = db.Column(db.Float, nullable=False)
    A10 = db.Column(db.Float, nullable=False)
    result = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), default=jakarta_time)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    __table_args__ = (db.PrimaryKeyConstraint('id', 'user_id'),)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load Model Machine Learning yang sudah disimpan
model = joblib.load('model.pkl')

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Ambil input dari form
        input_features = [float(request.form.get(f'A{i}')) for i in range(1, 11)]
    except Exception as e:
        return f"Terjadi kesalahan pada input: {e}"
    
    final_features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(final_features)
    result = int(prediction[0])
    
    # Menghitung local id untuk user
    max_id = db.session.query(db.func.max(Prediction.id)).filter(Prediction.user_id == current_user.id).scalar() or 0
    new_local_id = max_id + 1

    new_prediction = Prediction(
        id=new_local_id,
        A1=input_features[0],
        A2=input_features[1],
        A3=input_features[2],
        A4=input_features[3],
        A5=input_features[4],
        A6=input_features[5],
        A7=input_features[6],
        A8=input_features[7],
        A9=input_features[8],
        A10=input_features[9],
        result=result,
        user_id=current_user.id
    )
    db.session.add(new_prediction)
    db.session.commit()
    
    additional_text = ""
    if result in [3, 4]:
        additional_text = " Disarankan membawa anak anda pada specialist."
    
    prediction_text = f'Prediksi Tingkat Keparahan Anak: {result}{additional_text}'
    
    return render_template('index.html', prediction_text=prediction_text)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login berhasil!', 'login_success')
            return redirect(url_for('home'))
        else:
            flash('Username atau password salah.', 'login_error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username sudah digunakan.', 'register_error')
            return redirect(url_for('register'))
        
        new_user = User(
            username=username,
            password=generate_password_hash(password, method='pbkdf2:sha256')
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Akun berhasil dibuat! Silakan login.', 'register_success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Anda telah logout.', 'logout_success')
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)
    
@app.route('/delete_history', methods=['POST'])
@login_required
def delete_history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    for pred in predictions:
        db.session.delete(pred)
    db.session.commit()
    flash("History prediksi berhasil dihapus.", "delete_success")
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)
