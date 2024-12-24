# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy as sp


# Başlık
st.title("Health Insurance Cost Prediction")

# Dosyayı yükleyip gösterme
try:
    df = pd.read_csv('insurance.csv')
    st.write("Number of rows and columns in the data set:", df.shape)

    # Verilerin ilk birkaç satırını göster
    st.subheader("Top few rows of the dataset:")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("The file 'insurance.csv' was not found. Please ensure it's in the same directory as this script.")

st.title("BMI vs Insurance Charges Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
sns.lmplot(x='bmi', y='charges', data=df, aspect=2, height=6)
plt.xlabel('Body Mass Index $(kg/m^2)$: as Independent variable')
plt.ylabel('Insurance Charges: as Dependent variable')
plt.title('Charge Vs BMI')
st.pyplot(plt)


st.title("Correlation Heatmap")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
# Grafik oluşturma
corr = numeric_df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, cmap='Wistia', annot=True)

# Grafiği Streamlit'te gösterme
st.pyplot(fig)


# Başlık
st.title("Distribution of Insurance Charges")

# Grafik oluşturma
f = plt.figure(figsize=(12, 4))

# Birinci grafik: Sigorta ücretlerinin dağılımı
ax = f.add_subplot(121)
sns.histplot(df['charges'], bins=50, color='r', ax=ax, kde=True)
ax.set_title('Distribution of Insurance Charges')

# İkinci grafik: Log ölçeğinde sigorta ücretlerinin dağılımı
ax = f.add_subplot(122)
sns.histplot(np.log10(df['charges']), bins=40, color='b', ax=ax, kde=True)
ax.set_title('Distribution of Insurance Charges in $log$ Scale')
ax.set_xscale('log')

# Grafikleri Streamlit'te gösterme
st.pyplot(f)

st.title("Violin Plot Analysis of Insurance Charges")


# Grafik oluşturma
f = plt.figure(figsize=(14, 6))

# Birinci grafik: Cinsiyete göre ücretlerin violin grafiği
ax = f.add_subplot(121)
sns.violinplot(x='sex', y='charges', data=df, palette='Wistia', ax=ax)
ax.set_title('Violin Plot of Charges vs Sex')

# İkinci grafik: Sigara kullanımı durumuna göre ücretlerin violin grafiği
ax = f.add_subplot(122)
sns.violinplot(x='smoker', y='charges', data=df, palette='magma', ax=ax)
ax.set_title('Violin Plot of Charges vs Smoker')

# Grafikleri Streamlit'te gösterme
st.pyplot(f)
st.write("""
**Sol grafik:** Erkek ve kadınlar için sigorta ücretlerinin yaklaşık olarak aynı aralıkta olduğu görülüyor, ortalama ücret 5000 dolar civarındadır. 

**Sağ grafik:** Sigara içenler için sigorta ücreti, sigara içmeyenlere kıyasla çok daha geniş bir aralıkta yer alıyor. Sigara içmeyenlerin ortalama ücreti yaklaşık olarak 5000 dolarken, sigara içenler için en düşük sigorta ücreti 5000 dolardır.
""")

st.title("Box Plot of Charges vs Children")


# Grafik oluşturma
plt.figure(figsize=(14, 6))
sns.boxplot(x='children', y='charges', hue='sex', data=df, palette='rainbow')
plt.title('Box Plot of Charges vs Children')

# Grafiği Streamlit'te gösterme
st.pyplot()

# Yorumlar
st.write("""
**Grafikteki Yorumlar:**
- Cocuk sayısı arttıkça, sigorta ücretlerinin genellikle arttığı gözlemleniyor. 
- Hem erkekler hem de kadınlar için benzer ücret dağılımları gözükmekte.
""")




# İstatistiksel Yorumlar
# Verileri gruplandırarak yorum yapalım
region_stats = df.groupby('region')['charges'].describe()

# Yorumları Streamlit'te gösterme
st.write("### İstatistiksel Yorumlar:")

# Bölge bazında sigorta ücretlerinin istatistiksel özetini gösterme
st.write(region_stats)

# Ortalama ücretler için yorum
avg_charges = df.groupby('region')['charges'].mean()

# En yüksek ve en düşük ücretlerin olduğu bölgeyi bulma
max_charge_region = avg_charges.idxmax()
min_charge_region = avg_charges.idxmin()

# Ortalama ücretlerin bulunduğu bölgeyi yazdırma
st.write(f"En yüksek ortalama sigorta ücreti: {max_charge_region} bölgesinde.")
st.write(f"En düşük ortalama sigorta ücreti: {min_charge_region} bölgesinde.")

# Farklı cinsiyetler arasında ücret farklarını yazdırma
charge_diff = df[df['sex'] == 'male'].groupby('region')['charges'].mean() - df[df['sex'] == 'female'].groupby('region')['charges'].mean()

st.write("### Cinsiyetler Arasındaki Ücret Farkları:")
st.write(charge_diff)

# Başlık
st.title("Scatter Plots of Charges vs Age and Charges vs BMI")



# Grafik oluşturma
f = plt.figure(figsize=(14, 6))

# Scatter plot for Charges vs Age
ax = f.add_subplot(121)
sns.scatterplot(x='age', y='charges', data=df, palette='magma', hue='smoker', ax=ax)
ax.set_title('Scatter plot of Charges vs Age')

# Scatter plot for Charges vs BMI
ax = f.add_subplot(122)
sns.scatterplot(x='bmi', y='charges', data=df, palette='viridis', hue='smoker', ax=ax)
ax.set_title('Scatter plot of Charges vs BMI')

# Grafiği Streamlit'te gösterme
st.pyplot(f)

# İstatistiksel Yorumlar
# Yaş ve sigorta ücreti arasındaki ilişki
age_stats = df.groupby('smoker')['age'].mean()
charge_by_age = df.groupby('age')['charges'].mean()

# Ortalama yaş ve sigorta ücreti
avg_age_smoker = age_stats['yes']
avg_age_non_smoker = age_stats['no']

# Yaş aralıklarıyla sigorta ücreti analizi
charge_by_age_range = df.groupby(pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 70, 80]))['charges'].mean()

# BMI ve sigorta ücreti arasındaki ilişki
charge_by_bmi = df.groupby(pd.cut(df['bmi'], bins=[10, 20, 25, 30, 35, 40, 45]))['charges'].mean()

# Yaş ve sigorta ücreti arasındaki ilişkinin yorumlanması
st.write(f"**Yaş ve Sigorta Ücreti İlişkisi: **")
st.write(f"Sigara içenlerin ortalama yaşı: {avg_age_smoker:.2f}")
st.write(f"Sigara içmeyenlerin ortalama yaşı: {avg_age_non_smoker:.2f}")
st.write("Yaş arttıkça sigorta ücretleri de artıyor. Sigara içenlerin sigorta ücretleri, sigara içmeyenlerden daha yüksek.")

# Yaş aralıklarıyla sigorta ücreti
st.write("**Yaş Aralıklarına Göre Sigorta Ücretleri:**")
st.write(charge_by_age_range)

# BMI ve sigorta ücreti ile ilgili yorumlar
st.write("**BMI ve Sigorta Ücreti İlişkisi:**")
st.write(charge_by_bmi)
st.write("BMI arttıkça sigorta ücretlerinde de bir artış gözlemleniyor, özellikle BMI değeri 30'un üzerine çıkan bireylerde sigorta ücretleri daha yüksek.")

# Dummy variable
categorical_columns = ['sex','children', 'smoker', 'region']
df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')

## Log transform
df_encode['charges'] = np.log(df_encode['charges'])


X = df_encode.drop('charges',axis=1) # Independet variable
y = df_encode['charges'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)

# Step 1: add x0 =1 to dataset
X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

# Step2: build model
theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,y_train))
# The parameters for linear regression model
parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
columns = ['intersect:x_0=1'] + list(X.columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})

# Scikit Learn module

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

#Parameter
sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
parameter_df

# Normal Equation ile Tahmin
y_pred_norm = np.matmul(X_test_0, theta)

# Mean Square Error (MSE) Hesaplama
J_mse = np.sum((y_pred_norm - y_test)**2) / X_test_0.shape[0]

# R Kare Hesaplama
sse = np.sum((y_pred_norm - y_test)**2)
sst = np.sum((y_test - y_test.mean())**2)
R_square = 1 - (sse / sst)

# Sonuçları Streamlit'te Gösterme
st.write(f"The Mean Square Error (MSE) or J(θ) is: {J_mse:.4f}")
st.write(f"R square obtained for the normal equation method is: {R_square:.4f}")

# sklearn regression module
y_pred_sk = lin_reg.predict(X_test)

#Evaluvation: MSE

J_mse_sk = mean_squared_error(y_pred_sk, y_test)

# R_square
R_square_sk = lin_reg.score(X_test,y_test)

st.write(f"The Mean Square Error(MSE) or J(theta) is: {J_mse_sk:.4f}")
st.write(f"R square obtain for scikit learn library is : {R_square_sk:.4f}")

# Başlık
st.title("Linearity Check and Residual Error Evaluation")

# Linearlık Kontrolü
f = plt.figure(figsize=(14, 5))

# Gerçek ve Tahmin Edilen Değerlerin Karşılaştırılması (Linearity Check)
ax = f.add_subplot(121)
sns.scatterplot(x=y_test, y=y_pred_sk, ax=ax, color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Residual Hataların Normalite ve Ortalamalarının Kontrolü
ax = f.add_subplot(122)
sns.histplot((y_test - y_pred_sk), ax=ax, color='b', kde=True)
ax.axvline((y_test - y_pred_sk).mean(), color='k', linestyle='--')
ax.set_title('Check for Residual Normality & Mean: \n Residual Error')

# Streamlit'te Grafiği Gösterme
st.pyplot(f)

# Başlık
st.title("Multivariate Normality and Homoscedasticity Check")


# Multivariate Normality Check (Q-Q Plot)
f, ax = plt.subplots(1, 2, figsize=(14, 6))

# Q-Q Plot
_, (_, _, r) = sp.stats.probplot((y_test - y_pred_sk), fit=True, plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')

# Homoscedasticity Check (Residual vs Predicted)
sns.scatterplot(y=(y_test - y_pred_sk), x=y_pred_sk, ax=ax[1], color='r')
ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted')

# Streamlit'te Grafiği Gösterme
st.pyplot(f)

st.write("Q-Q Plot: Modelin tahmin ettiği değerlerin gerçek değerlerle olan farklarının normal dağılıma uygun olup olmadığını kontrol eder.")
st.write("Residual vs Predicted: Kalıntıların (residuals) tahmin edilen değerlerle ilişkisini kontrol eder, homoscedasticity olup olmadığını belirler.")
