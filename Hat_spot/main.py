#importing necessary libraries
import pandas as pd
import numpy as np 
from esm_embeddings import get_esm_embeddings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
from model import hatspot_mlp
from torch import nn
from training import train_dual_input_model
from training import plot_training_metrics
from evaluate import get_predictions
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

#grabbing csv file
print("Loading data...")
dataframe = pd.read_excel(r"C:\Users\Owner\Desktop\nih\media-12.xlsx", sheet_name='Supplementary Table 4')
print("Data loaded successfully!")

#cleaning data
print("Cleaning data...")
#removing rows with missing values
dataframe = dataframe.dropna()

#removing rows that have values that are considered "non-existant Kd values"
df_filtered = dataframe[
    (dataframe['BRD7'] <= 24000) | 
    (dataframe['P300 KIX'] <= 100400) | 
    (dataframe['P300 TAZ1'] <= 53000) | 
    (dataframe['P300 TAZ2'] <= 43000) | 
    (dataframe['P300'] <= 18000)
]

# Reset index after filtering to avoid KeyError
dataframe = df_filtered.reset_index(drop=True)
print("Data cleaned successfully!")

#grabbing sequences only to get ESM values
print("Extracting sequences...")
total_data = dataframe.drop(['BRD7', 'P300', 'P300 KIX', 'P300 NCBD', 'P300 TAZ1',
       'P300 TAZ2', 'TFIID TAF12', 'PCAF TAF6L', 'P300 top binder',
       'P300 TAZ2 top binder', 'P300 KIX top binder', 'P300 TAZ1 top binder',
       'P300 NCBD top binder', 'BRD7 top binder', 'TFIID TAF12 top binder',
       'TAF6L top binder', 'P300 binder', 'P300 KIX binder',
       'P300 TAZ2 binder', 'P300 TAZ1 binder', 'P300 NCBD binder',
       'BRD7 binder', 'TAF6L binder', 'TFIID TAF12 binder',
       'NCPR', 'Q Count', 'S Count', 'P Count', 'DE Count'], axis=1)

domain_data = {
    'Variant_ID': ['BRD7', 'P300', 'P300 KIX', 'P300 TAZ1', 'P300 TAZ2'],
    'Domain Sequence': ['MGKKHKKHKSDKHLYEEYVEKPLKLVLKVGGNEVTELSTGSSGHDSSLFEDKNDHDKHKDRKRKKRKKGEKQIPGEEKGRKRRRVKEDKKKRDRDRVENEAEKDLQCHAPVRLDLPPEKPLTSSLAKQEEVEQTPLQEALNQLMRQLQRKDPSAFFSFPVTDFIAPGYSMIIKHPMDFSTMKEKIKNNDYQSIEELKDNFKLMCTNAMIYNKPETIYYKAAKKLLHSGMKILSQERIQSLKQSIDFMADLQKTRKQKDGTDTSQSGEDGGCWQREREDSGDAEAHAFKSPSKENKKKDKDMLEDKFKSNNLEREQEQLDRIVKESGGKLTRRLVNSQCEFERRKPDGTTTLGLLHPVDPIVGEPGYCPVRLGMTTGRLQSGVNTLQGFKEDKRNKVTPVLYLNYGPYSSYAPHYDSTFANISKDDSDLIYSTYGEDSDLPSDFSIHEFLATCQDYPYVMADSLLDVLTKGGHSRTLQEMEMSLPEDEGHTRTLDTAKEMEITEVEPPGRLDSSTQDRLIALKAVTNFGVPVEVFDSEEAEIFQKKLDETTRLLRELQEAQNERLSTRPPPNMICLLGPSYREMHLAEQVTNNLKELAQQVTPGDIVSTYGVRKAMGISIPSPVMENNFVDLTEDTEEPKKTDVAECGPGGS',
                        'MAENVVEPGPPSAKRPKLSSPALSASASDGTDFGSLFDLEHDLPDELINSTELGLTNGGDINQLQTSLGMVQDAASKHKQLSELLRSGSSPNLNMGVGGPGQVMASQAQQSSPGLGLINSMVKSPMTQAGLTSPNMGMGTSGPNQGPTQSTGMMNSPVNQPAMGMNTGMNAGMNPGMLAAGNGQGIMPNQVMNGSIGAGRGRQNMQYPNPGMGSAGNLLTEPLQQGSPQMGGQTGLRGPQPLKMGMMNNPNPYGSPYTQNPGQQIGASGLGLQIQTKTVLSNNLSPFAMDKKAVPGGGMPNMGQQPAPQVQQPGLVTPVAQGMGSGAHTADPEKRKLIQQQLVLLLHAHKCQRREQANGEVRQCNLPHCRTMKNVLNHMTHCQSGKSCQVAHCASSRQIISHWKNCTRHDCPVCLPLKNAGDKRNQQPILTGAPVGLGNPSSLGVGQQSAPNLSTVSQIDPSSIERAYAALGLPYQVNQMPTQPQVQAKNQQNQQPGQSPQGMRPMSNMSASPMGVGVQTPSLLSDSMLHSAINSQNPMMSENASVPSLGPMPTAAQPSTTGIRKQWHEDITQDLRNHLVHKLVQAIFPTPDPAALKDRRMENLVAYARKVEGDMYESANNRAEYYHLLAEKIYKIQKELEEKRRTRLQKQNMLPNAAGMVPVSMNPGPNMGQPQPGMTSNGPLPDPSMIRGSVPNQMMPRITPQSGLNQFGQMSMAQPPIVPRQTPPLQHHGQLAQPGALNPPMGYGPRMQQPSNQGQFLPQTQFPSQGMNVTNIPLAPSSGQAPVSQAQMSSSSCPVNSPIMPPGSQGSHIHCPQLPQPALHQNSPSPVPSRTPTPHHTPPSIGAQQPPATTIPAPVPTPPAMPPGPQSQALHPPPRQTPTPPTTQLPQQVQPSLPAAPSADQPQQQPRSQQSTAASVPTPTAPLLPPQPATPLSQPAVSIEGQVSNPPSTSSTEVNSQAIAEKQPSQEVKMEAKMEVDQPEPADTQPEDISESKVEDCKMESTETEERSTELKTEIKEEEDQPSTSATQSSPAPGQSKKKIFKPEELRQALMPTLEALYRQDPESLPFRQPVDPQLLGIPDYFDIVKSPMDLSTIKRKLDTGQYQEPWQYVDDIWLMFNNAWLYNRKTSRVYKYCSKLSEVFEQEIDPVMQSLGYCCGRKLEFSPQTLCCYGKQLCTIPRDATYYSYQNRYHFCEKCFNEIQGESVSLGDDPSQPQTTINKEQFSKRKNDTLDPELFVECTECGRKMHQICVLHHEIIWPAGFVCDGCLKKSARTRKENKFSAKRLPSTRLGTFLENRVNDFLRRQNHPESGEVTVRVVHASDKTVEVKPGMKARFVDSGEMAESFPYRTKALFAFEEIDGVDLCFFGMHVQEYGSDCPPPNQRRVYISYLDSVHFFRPKCLRTAVYHEILIGYLEYVKKLGYTTGHIWACPPSEGDDYIFHCHPPDQKIPKPKRLQEWYKKMLDKAVSERIVHDYKDIFKQATEDRLTSAKELPYFEGDFWPNVLEESIKELEQEEEERKREENTSNESTDVTKGDSKNAKKKNNKKTSKNKSSLSRGNKKKPGMPNVSNDLSQKLYATMEKHKEVFFVIRLIAGPAANSLPPIVDPDPLIPCDLMDGRDAFLTLARDKHLEFSSLRRAQWSTMCMLVELHTQSQDRFVYTCNECKHHVETRWHCTVCEDYDLCITCYNTKNHDHKMEKLGLGLDDESNNQQAAATQSPGDSRRLSIQRCIQSLVHACQCRNANCSLPSCQKMKRVVQHTKGCKRKTNGGCPICKQLIALCCYHAKHCQENKCPVPFCLNIKQKLRQQQLQHRLQQAQMLRRRMASMQRTGVVGQQQGLPSPTPATPTTPTGQQPTTPQTPQPTSQPQPTPPNSMPPYLPRTQAAGPVSQGKAAGQVTPPTPPQTAQPPLPGPPPAAVEMAMQIQRAAETQRQMAHVQIFQRPIQHQMPPMTPMAPMGMNPPPMTRGPSGHLEPGMGPTGMQQQPPWSQGGLPQPQQLQSGMPRPAMMSVAQHGQPLNMAPQPGLGQVGISPLKPGTVSQQALQNLLRTLRSPSSPLQQQQVLSILHANPQLLAAFIKQRAAKYANSNPQPIPGQPGMPQGQPGLQPPTMPGQQGVHSNPAMQNMNPMQAGVQRAGLPQQQPQQQLQPPMGGMSPQAQQMNMNHNTMPSQFRDILRRQQMMQQQQQQGAGPGIGPGMANHNQFQQPQGVGYPPQQQQRMQHHMQQMQQGNMGQIGQLPQALGAEAGASLQAYQQRLLQQQMGSPVQPNPMSPQQHMLPNQAQSPHLQGQQIPNSLSNQVRSPQPVPSPRPQSQPPHSSPSPRMQPQPSPHHVSPQTSSPHPGLVAAQANPMEQGHFASPDQNSMLSQLASNPGMANLHGASATDLGLSTDNSDLNSNLSQSTLDIH',
                        'GIRKQWHEDITQDLRNHLVHKLVQAIFPTPDPAALKDRRMENLVAYARKVEGDMYESANNRAEYYHLLAEKIYKIQKELE',
                        'DPEKRKLIQQQLVLLLHAHKCQRREQANGEVRQCNLPHCRTMKNVLNHMTHCQSGKSCQVAHCASSRQIISHWKNCTRHDCPVCLPL',
                        'GDSRRLSIQRCIQSLVHACQCRNANCSLPSCQKMKRVVQHTKGCKRKTNGGCPICKQLIALCCYHAKHCQENKCPVPFCLNI',
                        ]
}

domain_df = pd.DataFrame(domain_data)
sequence_df = pd.concat([total_data, domain_df], ignore_index=True)
print("Sequences extracted successfully!")

#grabbing ESM values
print("Extracting ESM values...")
data_list = list(zip(sequence_df['Variant_ID'], sequence_df['Domain Sequence']))
embeddings = get_esm_embeddings(data_list, batch_size=8)
print("ESM values extracted successfully!")

#saving ESM embeddings
print("Saving ESM embeddings...")
np.save('esm_embeddings.npy', embeddings)
print("ESM embeddings saved successfully!")

#Creating the dataset 
print("Creating dataset...")

#combining ESM embeddings with the cleaned data
embeddings_df = pd.DataFrame(embeddings)
binder_protein = ['BRD7', 'P300', 'P300 KIX', 'P300 TAZ1', 'P300 TAZ2']
df_final_1 = []

for idx in range(len(dataframe)):
    variant_id = dataframe.iloc[idx]['Variant_ID']
    domain_embedding = embeddings_df.iloc[idx]['pooled_embedding']

    for binder in binder_protein:
        binder_idx = len(dataframe) + binder_protein.index(binder)
        binder_embedding = embeddings_df.iloc[binder_idx]['pooled_embedding']
        kd_value = dataframe.iloc[idx][binder]
        
        df_final_1.append({
            'Variant_ID': variant_id,
            'domain_pooled': domain_embedding,
            'Binder': binder,
            'Binder_pooled': binder_embedding,
            'Kd': kd_value
        })

df_final_1 = pd.DataFrame(df_final_1)
df_final_1.dropna(subset=['Kd'], inplace=True)

#dropping rows with Kd values -> we assume that these are non-binding numbers
df_final = df_final_1[
    ((df_final_1["Binder"] == "P300") & (df_final_1["Kd"] <= 18000)) |
    ((df_final_1["Binder"] == "BRD7") & (df_final_1["Kd"] <= 24000)) |
    ((df_final_1["Binder"] == "P300 KIX") & (df_final_1["Kd"] <= 100400)) |
    ((df_final_1["Binder"] == "P300 TAZ1") & (df_final_1["Kd"] <= 53000)) |
    ((df_final_1["Binder"] == "P300 TAZ2") & (df_final_1["Kd"] <= 43000))
].copy()

#due to high values we log base e the final Kd values
df_final['Kd'] = np.log(df_final['Kd'])

print(f"Final dataframe shape: {df_final.shape}")
print("Dataset created successfully!")

#scaling data
print("Preparing data to split for training, testing, and validating...")

# First split: 80% train+val, 20% test (stratified by binder)
train_val_data, test_data = train_test_split(
    df_final,
    test_size=0.2,
    random_state=42,
    stratify=df_final['Binder']
)

# Second split: 75% train, 25% val (stratified by binder)
train_data, validate_data = train_test_split(
    train_val_data,
    test_size=0.25,
    random_state=42,
    stratify=train_val_data['Binder']
)

# Prepare data
X_train_domain = np.stack(train_data['domain_pooled'].values)
X_train_binder = np.stack(train_data['Binder_pooled'].values)
y_train = train_data['Kd'].values

X_val_domain = np.stack(validate_data['domain_pooled'].values)
X_val_binder = np.stack(validate_data['Binder_pooled'].values)
y_val = validate_data['Kd'].values

X_test_domain = np.stack(test_data['domain_pooled'].values)
X_test_binder = np.stack(test_data['Binder_pooled'].values)
y_test = test_data['Kd'].values

print(f"X_train_domain shape: {X_train_domain.shape}")
print(f"X_train_binder shape: {X_train_binder.shape}")

print("Scaling data...")
# Scale each input separately
scaler_domain = StandardScaler()
X_train_domain_scaled = scaler_domain.fit_transform(X_train_domain)
X_val_domain_scaled = scaler_domain.transform(X_val_domain)
X_test_domain_scaled = scaler_domain.transform(X_test_domain)

scaler_binder = StandardScaler()
X_train_binder_scaled = scaler_binder.fit_transform(X_train_binder)
X_val_binder_scaled = scaler_binder.transform(X_val_binder)
X_test_binder_scaled = scaler_binder.transform(X_test_binder)

# Scale y
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

#saving the StandardScaler objects
joblib.dump(scaler_domain, 'scaler_domain.pkl')
joblib.dump(scaler_binder, 'scaler_binder.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Convert to PyTorch tensors
X_train_domain_tensor = torch.FloatTensor(X_train_domain_scaled)
X_train_binder_tensor = torch.FloatTensor(X_train_binder_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1)

X_val_domain_tensor = torch.FloatTensor(X_val_domain_scaled)
X_val_binder_tensor = torch.FloatTensor(X_val_binder_scaled)
y_val_tensor = torch.FloatTensor(y_val_scaled).reshape(-1, 1)

X_test_domain_tensor = torch.FloatTensor(X_test_domain_scaled)
X_test_binder_tensor = torch.FloatTensor(X_test_binder_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_domain_tensor, X_train_binder_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val_domain_tensor, X_val_binder_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Data prepared successfully!")

#initializing model
print("Initializing model...")
domain_input_size = X_train_domain.shape[1]
binder_input_size = X_train_binder.shape[1]
hidden_size = 16
output_size = 1

model = hatspot_mlp(
    domain_input_size=domain_input_size,
    tf_input_size=binder_input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    dropout_rate=0.5
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

#training model
print("Training model...")
results = train_dual_input_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=300,
    lr=0.0005,
    weight_decay=5e-3,
    l1_lambda=5e-3,
    patience=25,
    print_every=10,
    save_plot=True,
    plot_filename='training_metrics.png'
)

# Access results
trained_model = results['model']
best_val_loss = results['best_val_loss']

# Plot metrics
plot_training_metrics(
    results['train_losses'],
    results['val_losses'],
    results['train_maes'],
    results['val_maes'],
    results['train_r2s'],
    results['val_r2s'],
    results['best_val_loss'],
    filename='my_custom_plot.png'
)

#saving trained model
print("Saving trained model...")
torch.save(trained_model.state_dict(), 'trained_model.pth')

#evaluating the model
print("Evaluating model...")
train_results = get_predictions(
    model,
    X_train_domain_tensor,
    X_train_binder_tensor,
    y_train,
    train_data
)

val_results = get_predictions(
    model,
    X_val_domain_tensor,
    X_val_binder_tensor,
    y_val,
    validate_data
)

test_results = get_predictions(
    model,
    X_test_domain_tensor,
    X_test_binder_tensor,
    y_test,
    test_data
)

# Add dataset labels
train_results['Dataset'] = 'Train'
val_results['Dataset'] = 'Validation'
test_results['Dataset'] = 'Test'

# Combine results
all_results = pd.concat([train_results, val_results, test_results], ignore_index=True)

# Save results
all_results.to_csv('model_predictions.csv', index=False)
print("Model predictions saved to 'model_predictions.csv'!")

# Calculate metrics for each dataset
def calculate_metrics(df):
    """Calculate R2, MAE, and RMSE metrics"""
    r2 = r2_score(df['Kd'], df['Predicted_log'])
    mae = mean_absolute_error(df['Kd'], df['Predicted_log'])
    rmse = np.sqrt(mean_squared_error(df['Kd'], df['Predicted_log']))
    return r2, mae, rmse

train_r2, train_mae, train_rmse = calculate_metrics(train_results)
val_r2, val_mae, val_rmse = calculate_metrics(val_results)
test_r2, test_mae, test_rmse = calculate_metrics(test_results)

# Add residuals to results
all_results['Residual'] = all_results['Kd'] - all_results['Predicted_log']

# Create comprehensive scatter plots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: All data combined
ax = axes[0, 0]
for dataset, color in [('Train', 'blue'), ('Validation', 'orange'), ('Test', 'green')]:
    data = all_results[all_results['Dataset'] == dataset]
    ax.scatter(data['Kd'], data['Predicted_log'],
               alpha=0.5, s=50, label=dataset, color=color)

min_val = all_results['Kd'].min()
max_val = all_results['Kd'].max()
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Kd', fontsize=12)
ax.set_ylabel('Predicted_log Kd', fontsize=12)
ax.set_title('All Datasets Combined', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Training set
ax = axes[0, 1]
ax.scatter(train_results['Kd'], train_results['Predicted_log'],
           alpha=0.6, s=50, color='blue')
ax.plot([train_results['Kd'].min(), train_results['Kd'].max()],
        [train_results['Kd'].min(), train_results['Kd'].max()],
        'r--', linewidth=2)
ax.set_xlabel('Actual Kd', fontsize=12)
ax.set_ylabel('Predicted_log Kd', fontsize=12)
ax.set_title(f'Training Set\nR²={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Validation set
ax = axes[1, 0]
ax.scatter(val_results['Kd'], val_results['Predicted_log'],
           alpha=0.6, s=50, color='orange')
ax.plot([val_results['Kd'].min(), val_results['Kd'].max()],
        [val_results['Kd'].min(), val_results['Kd'].max()],
        'r--', linewidth=2)
ax.set_xlabel('Actual Kd', fontsize=12)
ax.set_ylabel('Predicted_log Kd', fontsize=12)
ax.set_title(f'Validation Set\nR²={val_r2:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 4: Test set
ax = axes[1, 1]
ax.scatter(test_results['Kd'], test_results['Predicted_log'],
           alpha=0.6, s=50, color='green')
ax.plot([test_results['Kd'].min(), test_results['Kd'].max()],
        [test_results['Kd'].min(), test_results['Kd'].max()],
        'r--', linewidth=2)
ax.set_xlabel('Actual Kd', fontsize=12)
ax.set_ylabel('Predicted_log Kd', fontsize=12)
ax.set_title(f'Test Set\nR²={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
print("Scatter plots saved to 'prediction_scatter_plots.png'!")

# Residual plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (dataset, data, color) in enumerate([
    ('Train', train_results, 'blue'),
    ('Validation', val_results, 'orange'),
    ('Test', test_results, 'green')
]):
    ax = axes[idx]
    residuals = data['Kd'] - data['Predicted_log']
    ax.scatter(data['Predicted_log'], residuals, alpha=0.6, s=50, color=color)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted_log Kd', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted_log)', fontsize=12)
    ax.set_title(f'{dataset} Residuals', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
print("Residual plots saved to 'residual_plots.png'!")

# Print summary statistics
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"\nTraining Set (n={len(train_results)}):")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MAE:      {train_mae:.4f}")
print(f"  RMSE:     {train_rmse:.4f}")

print(f"\nValidation Set (n={len(val_results)}):")
print(f"  R² Score: {val_r2:.4f}")
print(f"  MAE:      {val_mae:.4f}")
print(f"  RMSE:     {val_rmse:.4f}")

print(f"\nTest Set (n={len(test_results)}):")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE:      {test_mae:.4f}")
print(f"  RMSE:     {test_rmse:.4f}")
print("="*60)

print("\nAll outputs saved successfully!")
