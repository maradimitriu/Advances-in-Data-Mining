# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random

# 1. COSINE SIMILARITY:

def cos_similarity_formula(vec_i, vec_j):
    # Compute cosine similarity between row i and row j
    dot_product = np.dot(vec_i, vec_j)
    norm_i = np.linalg.norm(vec_i)
    norm_j = np.linalg.norm(vec_j)
    if norm_i == 0 or norm_j == 0:
        cosine_similarity = 0
    else:
        cosine_similarity = dot_product / (norm_i * norm_j)
    cosine_similarity = max(min(cosine_similarity, 1.0), -1.0) # to avoid values higher than 1

    return cosine_similarity


def similarity_matrix(matrix, k=5, axis=0):
    """
    This function should contain the code to compute the cosine similarity
    (according to the formula seen at the lecture) between users (axis=0) or 
    items (axis=1) and return a dictionary where each key represents a user
    (or item) and the value is a list of the top k most similar users or items,
    along with their similarity scores.
    
    Args:
        matrix (pd.DataFrame) : user-item rating matrix (df)
        k (int): number of top k similarity rankings to return for each \
                    entity (default=5)
        axis (int): 0: calculate similarity scores between users \
                        (rows of the matrix), 
                    1: claculate similarity scores between items \
                        (columns of the matrix)
    
    Returns:
        similarity_dict (dictionary): dictionary where the keys are users 
        (or items) and the values are lists of tuples containing the most 
        similar users (or items) along with their similarity scores.

    Note that is NOT allowed to authomatically compute cosine similarity using
    an apposite function from any package, the computation should follow the 
    formula that has been discussed during the lecture and that can be found in
    the slides.

    Note that is allowed to convert the DataFrame into a Numpy array for 
    faster computation.
    """

    # Check that axis is valid
    if axis not in [0,1]:
        raise ValueError("Axis must be 0 (users) or 1 (items)")
    if axis==1:
        matrix = matrix.T # if axis is 1, we transpose the matrix to make items as rows

    row_labels = matrix.index.tolist() # Save row labels (user or item IDs)
    matrix = matrix.to_numpy()  # Convert DataFrame to Numpy array for faster computation
    similarity_dict= {} # Initialize dictionary for output

    # TO DO: Handle the absence of ratings (missing values in the matrix)
    for i in range(matrix.shape[0]): # loop through number of rows (1 for cols)
        similarity_dict[row_labels[i]] = []
        row_i = matrix[i,:]
        for j in range (i+1, matrix.shape[0]): # loop through the rest of the rows
            similarity_dict[row_labels[j]] = []
            row_j = matrix[j,:]
            # Create mask to deal with NaN
            mask = ~np.isnan(row_i) & ~np.isnan(row_j) # ~ is negation
            # Apply mask
            vec_i = row_i[mask]
            vec_j = row_j[mask]
            # Add to dictionary
            similarity_dict[row_labels[i]].append((row_labels[j], cos_similarity_formula(vec_i, vec_j)))
            similarity_dict[row_labels[j]].append((row_labels[i], cos_similarity_formula(vec_i, vec_j)))

    # Sort similarity lists and keep only top k
    for key in similarity_dict:
        similarity_dict[key] = sorted(
            similarity_dict[key],
            key=lambda x: x[1],    # define a small function that returns the second elem of the tuple (similarity score)
            reverse=True           # highest first
        )[:k]  # keep top-k

    return similarity_dict


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictonary containing user similarities, \
            obtained using the similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to user-based \
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target user
    # Make a list with the scores of user_id from the dictionary
    top_similarity_scores = [score for user,score in user_similarity[user_id]]
    top_users = [user for user, score in user_similarity[user_id]]

    # TO DO: implement user-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0

    for i in range(k):
        rating = user_item_matrix.loc[top_users[i], movie_id]
        if not np.isnan(rating):
            numerator = numerator + top_similarity_scores[i] * rating
            denominator = denominator + top_similarity_scores[i]

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative
    filtering, returning the predicted rate associated to a target user-movie 
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, \
            obtained using the similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to item-based 
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target item

    # Make a list with the scores of user_id from the dictionary
    top_similarity_scores = [score for user,score in item_similarity[movie_id]]
    top_users = [user for user, score in item_similarity[movie_id]]

    # TO DO: implement item-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0  

    for i in range(k):
        rating = user_item_matrix.loc[top_users[i], movie_id]
        if not np.isnan(rating):
            numerator = numerator + top_similarity_scores[i] * rating
            denominator = denominator + top_similarity_scores[i]

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating

# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,
        learning_rate=0.001,
        regularization=0.02,
        n_steps=2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function should contain the code to implement matrix factorisation
    using the Gradient Descent with Regularization method (according to the psuedo code
    seen at the lecture), returning the user and item matrices.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        feature_dimension (int): number of latent features (default=2)
        learning_rate (float): learning rate for gradient descent \
            (default=0.001)
        regularization (float): regularization parameter (default=0.02)
        n_steps (int): number of iterations for gradient descent \
            (default=2000)

    Returns:
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
    """

    user_matrix = np.random.rand(utility_matrix.shape[0], feature_dimension)
    item_matrix = np.random.rand(utility_matrix.shape[1], feature_dimension)

    for step in range(n_steps):
        # Implement the algorithm to update user_matrix and item_matrix
        for i in range(utility_matrix.shape[0]): # row loop
            for j in range(utility_matrix.shape[1]): # column loop
                rating = utility_matrix[i,j]
                if np.isnan(rating):   # Skip missing ratings
                    continue # go on with the next iteration of the loop
                prediction = np.dot(user_matrix[i, :], item_matrix[j, :]) # vector multiplication (dot product)
                err = rating - prediction
                user_matrix[i, :] += learning_rate * (err * item_matrix[j, :] - regularization * user_matrix[i, :])
                item_matrix[j, :] += learning_rate * (err * user_matrix[i, :] - regularization * item_matrix[j, :])

    return user_matrix, item_matrix

if __name__ == "__main__":
    path = r"G:\My Drive\masters_leiden\advances_in_data_mining\Advances-in-Data-Mining\Assign_1\u.data"       
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])
    df = df.pivot_table(
        index = 'UserID', 
        columns = 'MovieID', 
        values = 'Rating'
    )

    # You can use this section for testing the similarity_matrix function: 
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10,[]))

    
    # You can use this section for testing the user_based_cf and the 
    # item_based_cf functions: Return the predicted ratings assigned by user 
    # 13 to movie 100:
    user_id = 13  
    movie_id = 100  

    u_predicted_rating = user_based_cf(
        user_id, 
        movie_id, 
        user_similarity_matrix, 
        user_item_matrix = df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id, 
        item_similarity_matrix,
        user_item_matrix = df, 
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
        [5, 2, 4, 4, 3],
        [3, 1, 2, 4, 1],
        [2, np.nan, 3, 1, 4],
        [2, 5, 4, 3, 5],
        [4, 4, 5, 4, np.nan],
    ])
    user_matrix, item_matrix = matrix_factorization(
        utility_matrix, learning_rate=0.001, n_steps=5000
    )

    print("Current guess:\n", np.dot(user_matrix, item_matrix.T))
