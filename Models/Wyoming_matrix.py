# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.sparse import coo_matrix, csr_matrix


# %% [markdown]
# Read in standardized csv files and merge them into one Dataframe with relevant information

# %%
df_form = pd.read_csv("formationout.csv")
df_well = pd.read_csv("out.csv")
#Merge the 2 CSVs by API number
df_merged = df_well.merge(df_form, how = "left", on = "API Number")
#drop well number identifier since we are using API number
df_merged.drop(columns="Well Number", inplace = True)

# %% [markdown]
# Standardize depth around the minimum value

# %%
df_merged["Top MD"] = df_merged["Top MD"] - df_merged["Top MD"].min()

# %% [markdown]
# Taking a sample of the Dataframe to holdout 

# %%
df_holdout = df_merged.sample(frac=0.2, random_state=4242001)
#make list of API numbers that we held out
heldout_APIs = []
for i in df_holdout["API Number"]:
    heldout_APIs.append(i)
#now we need to go back to our original Dataframe and set the vals we are holding out to 0
df_merged_heldout = df_merged.copy()
np.random.seed(4242001)
#get 5 random formation aliases to hold out
h = np.random.randint(0, 59, 5)
#hold out these tops by setting them to NaN, but keep the rest of the data intact
df_merged_heldout[df_merged_heldout["API Number"].isin(heldout_APIs)]["Form Alias"].replace(h, float("NaN"), inplace=True)


# %% [markdown]
# Sort the held out values by API Number and Formation Alias

# %%
df_holdout.sort_values(by=["API Number", "Form Alias"], inplace=True)

# %% [markdown]
# Make a sparse matrix from the Dataframe with random formation aliases held out

# %%
D_df = df_merged_heldout.pivot_table("Top MD","Form Alias","API Number").fillna(0)
D_df

# %% [markdown]
# Trying different ways of normalizing R, demeaning and normalizing with SKLearn

# %%
mms = MinMaxScaler()
R = D_df.values
target_vals = df_holdout["Top MD"]
R_normalize = mms.fit_transform(R, target_vals)

# %% [markdown]
# Create binarized matrix with values of 1 where there are depth values in the sparse matrix R and values of 0 where there are not depth values in the sparse matrix R.

# %%
from sklearn.preprocessing import binarize
A = binarize(R)


# %% [markdown]
# This is the code that runs Alternating Least Squares factorization

# %%
#ALS factorization from 
# https://github.com/mickeykedia/Matrix-Factorization-ALS/blob/master/ALS%20Python%20Implementation.py
# here items are the formation and users are the well
def runALS(A, R, n_factors, n_iterations, lambda_):
    """
    Runs Alternating Least Squares algorithm in order to calculate matrix.
    :param A: User-Item Matrix with ratings
    :param R: User-Item Matrix with 1 if there is a rating or 0 if not
    :param n_factors: How many factors each of user and item matrix will consider
    :param n_iterations: How many times to run algorithm
    :param lambda_: Regularization parameter
    :return:
    """
    print("Initiating ")
    lambda_ = lambda_
    n_factors = n_factors
    n, m = A.shape
    n_iterations = n_iterations
    Users = 5 * np.random.rand(n, n_factors)
    Items = 5 * np.random.rand(n_factors, m)

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MSE_List = []

    print("Starting Iterations")
    for iter in range(n_iterations):
        for i, Ri in enumerate(R):
            Users[i] = np.linalg.solve(
                np.dot(Items, np.dot(np.diag(Ri), Items.T))
                + lambda_ * np.eye(n_factors),
                np.dot(Items, np.dot(np.diag(Ri), A[i].T)),
                ).T
        print(
            "Error after solving for User Matrix:",
            get_error(A, Users, Items, R),
            )

        for j, Rj in enumerate(R.T):
            Items[:, j] = np.linalg.solve(
                np.dot(Users.T, np.dot(np.diag(Rj), Users))
                + lambda_ * np.eye(n_factors),
                np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])),
                )
        print(
            "Error after solving for Item Matrix:",
             get_error(A, Users, Items, R),
            )

        MSE_List.append(get_error(A, Users, Items, R))
        print("%sth iteration is complete..." % iter)
    return Users, Items
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(range(1, len(MSE_List) + 1), MSE_List); plt.ylabel('Error'); plt.xlabel('Iteration')
    # plt.title('Python Implementation MSE by Iteration \n with %d formations and %d wells' % A.shape);
    # plt.savefig('Python MSE Graph.pdf', format='pdf')
    # plt.show()


# %% [markdown]
# Get the User and Item vectors from ALS

# %%
U, Vt = runALS(R, A, 20, 20, 0.1)

# %% [markdown]
# Construct our predicted Dataframe from our User and Item vectors

# %%
recommendations = np.dot(U, Vt)
recsys_df = pd.DataFrame(data = recommendations[0:, 0:], index = D_df.index,
                        columns = D_df.columns)
recsys_df.head()

# %% [markdown]
# Reshape the predicted Dataframe to a more easily accessible format

# %%
recsys_df_reshaped = recsys_df.T.reset_index()

# %% [markdown]
# Reshape our predictions and merge them into one final Dataframe showcasing the relevant information 

# %%
#reshape predictions
flat_preds = recsys_df.unstack().reset_index()
#merge predictions with our testing data set
merged_df = pd.merge(df_holdout, flat_preds,  how='left', left_on=['API Number','Form Alias'], right_on = ['API Number','Form Alias'])
merged_df.rename(columns={0:"Predicted Depth"}, inplace=True)
#Drop irrelevant columns to more clearly showcase results
final_df = merged_df.drop(columns={"Northing", "Easting", "Normalized TVD", "True Vertical Depth"})
#add signed error colum to indicate whether we are over or under predicting
final_df["signed_error"] = final_df["Top MD"] - final_df["Predicted Depth"]
final_df.dropna().head()

# %% [markdown]
# Now from our final Dataframe we can find an error metric to evaluate the performance of our model.

# %%
from sklearn.metrics import mean_absolute_error as MAE
MAE(final_df["Top MD"].dropna().values, final_df["Predicted Depth"].dropna().values)

# %% [markdown]
# Plot the recommended depths for all formations for the first 5 wells vs the actual depths

# %%
for i in range(5):
    plt.scatter(recsys_df.iloc[0:, i].values, D_df.iloc[0:, i].values) #plot predicted vs actual
    plt.xlabel('predicted depth')
    plt.ylabel('actual depth')
    plt.plot(np.arange(0,recsys_df.iloc[0:,i].max()))

# %%
