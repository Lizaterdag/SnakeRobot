import pandas as pd

def check_back_to_back_duplicates(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check for back-to-back duplicate rows
    duplicates = df[df['X_Position'].shift() == df['X_Position']]

    duplicates.to_csv("back_to_back_duplicates2.csv", index=False)
    
    # Print the result
    if not duplicates.empty:
        print("Back-to-back duplicate rows found based on 'X_Position':")
    else:
        print("No back-to-back duplicate rows found based on 'X_Position'.")

if __name__ == "__main__":
    file_path = 'output_3.csv'  
    check_back_to_back_duplicates(file_path)
    # Check for back-to-back duplicate rows based on 'X_Position' column
    

# import pandas as pd

# # Load the CSV file (update file name if needed)
# df = pd.read_csv("output_3.csv")

# # Check for back-to-back duplicates in 'X_Position'
# df["Back_to_Back_Duplicate"] = df["X_Position"] == df["X_Position"].shift(1)

# # Display only rows where there are back-to-back duplicates
# back_to_back_duplicates = df[df["Back_to_Back_Duplicate"]]

# # Save results to a new CSV file
# back_to_back_duplicates.to_csv("back_to_back_duplicates2.csv", index=False)

# # Print the number of back-to-back duplicates
# print(f"Found {len(back_to_back_duplicates)} back-to-back duplicates.")
