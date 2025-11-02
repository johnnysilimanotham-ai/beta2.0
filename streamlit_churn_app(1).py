# ==================== PAGE 3: DATA CLEANING (UPDATED) ====================
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning & Preparation")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first!")
        st.stop()
    
    if st.session_state.target_col is None:
        st.warning("⚠️ Please select a target column on the 'Upload Data' page first!")
        st.stop()
        
    target_col = st.session_state.target_col
    
    # Use cleaned data if available, otherwise start with raw
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean.copy()
        st.info(f"📋 Working with previously cleaned data. You can make additional changes below. Target: **{target_col}**")
    else:
        # Start with raw data, but run basic target normalization/drop_na for consistency
        df = st.session_state.df_raw.copy()
        
        # --- NEW: Pre-run target normalization for accurate cleaning stats/preview ---
        try:
             df = normalize_target(df, target_col)
             df = df.dropna(subset=[target_col])
        except Exception as e:
             st.error(f"Error during initial target normalization: {str(e)}")
             st.stop()
        # --- END NEW ---
        
        st.info(f"📋 Starting with raw data (Target {target_col} normalized and missing target rows dropped).")

    # Display Target Column (No longer a selectbox, just a display)
    st.subheader("⚙️ Target Column (Set on Upload Page)")
    st.write(f"The current target column is: **{target_col}**")
    
    if target_col not in df.columns:
        st.error(f"❌ Selected target column '{target_col}' is missing from the dataset. Please go back to 'Upload Data'.")
        st.stop()
    
    # Show current data quality issues
    st.subheader("🔍 Data Quality Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Missing values details
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        st.subheader("⚠️ Columns with Missing Values")
        st.dataframe(missing_df, width='stretch')
    
    # Granular cleaning options
    st.subheader("🛠️ Cleaning Operations")
    
    # --- NEW SECTION: Column Type Overview (For Transparency) ---
    with st.expander("📝 Column Type Overview", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.write(f"**Numeric Features ({len(num_cols)}):** {', '.join(num_cols)}")
        st.write(f"**Categorical Features ({len(cat_cols)}):** {', '.join(cat_cols)}")
    # --- END NEW SECTION ---
    
    with st.expander("🗑️ Remove Duplicate Rows", expanded=False):
        if df.duplicated().sum() > 0:
            st.write(f"Found **{df.duplicated().sum()}** duplicate rows")
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.session_state.df_clean = df
                st.success(f"✅ Removed duplicates! Now {len(df)} rows.")
                st.rerun()
        else:
            st.info("No duplicates found")
    
    with st.expander("🔢 Handle Missing Values (Column-by-Column)", expanded=True):
        if missing_df.empty:
            st.info("No missing values found!")
        else:
            st.write("💡 **Tip:** Different strategies work better for different features. Try different approaches and check visualizations!")
            
            cols_with_missing = missing_df['Column'].tolist()
            selected_col = st.selectbox("Select column to handle", cols_with_missing)
            
            if selected_col:
                col_data = df[selected_col]
                st.write(f"**{selected_col}**: {col_data.isnull().sum()} missing values ({(col_data.isnull().sum()/len(df)*100):.1f}%)")
                
                # Determine if numeric or categorical
                is_numeric = pd.api.types.is_numeric_dtype(col_data)
                
                if is_numeric:
                    strategy = st.radio(
                        f"Strategy for {selected_col}",
                        ["Keep missing (let model handle)", "Fill with Mean", "Fill with Median", "Fill with Zero", "Drop rows with missing"],
                        key=f"strategy_{selected_col}"
                    )
                else:
                    strategy = st.radio(
                        f"Strategy for {selected_col}",
                        ["Keep missing (let model handle)", "Fill with Most Frequent", "Fill with 'Unknown'", "Drop rows with missing"],
                        key=f"strategy_{selected_col}"
                    )
                
                if st.button(f"Apply to {selected_col}", type="primary"):
                    if strategy == "Keep missing (let model handle)":
                        st.info("✅ Keeping missing values. The model preprocessing will handle them.")
                    elif strategy == "Fill with Mean":
                        df[selected_col].fillna(df[selected_col].mean(), inplace=True)
                        st.success(f"✅ Filled {selected_col} with mean value: {df[selected_col].mean():.2f}")
                    elif strategy == "Fill with Median":
                        df[selected_col].fillna(df[selected_col].median(), inplace=True)
                        st.success(f"✅ Filled {selected_col} with median value: {df[selected_col].median():.2f}")
                    elif strategy == "Fill with Zero":
                        df[selected_col].fillna(0, inplace=True)
                        st.success(f"✅ Filled {selected_col} with zeros")
                    elif strategy == "Fill with Most Frequent":
                        most_frequent = df[selected_col].mode()[0]
                        df[selected_col].fillna(most_frequent, inplace=True)
                        st.success(f"✅ Filled {selected_col} with most frequent value: {most_frequent}")
                    elif strategy == "Fill with 'Unknown'":
                        df[selected_col].fillna('Unknown', inplace=True)
                        st.success(f"✅ Filled {selected_col} with 'Unknown'")
                    elif strategy == "Drop rows with missing":
                        before = len(df)
                        df = df.dropna(subset=[selected_col])
                        st.success(f"✅ Dropped {before - len(df)} rows with missing {selected_col}")
                    
                    st.session_state.df_clean = df
                    st.rerun()
    
    with st.expander("🎯 Normalize Target Column", expanded=False):
        st.write(f"Current target: **{target_col}**")
        if target_col in df.columns:
            st.write(f"Unique values: {df[target_col].unique()}")
            st.write(f"Value counts:\n{df[target_col].value_counts()}")
        
        if st.button("Normalize Target to 0/1", help="Converts target column values to 0 or 1, and drops rows where target is missing."):
            try:
                # This logic is also in the initial part of this page for consistency, but kept here for explicit user action
                df = normalize_target(df, target_col)
                df = df.dropna(subset=[target_col])
                st.session_state.df_clean = df
                st.success("✅ Target normalized and rows with missing target dropped!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # MODIFIED SECTION: Remove Unwanted Columns
    with st.expander("🗂️ Remove Unwanted Columns", expanded=False):
        st.write("Select columns to permanently remove from the dataset (e.g., ID columns, metadata, highly redundant features).")
        
        # Auto-detect initial suggestions based on 'id' keyword
        suggested_id_cols = [c for c in df.columns if 'id' in c.lower() and c != target_col]
        if suggested_id_cols:
             st.info(f"💡 **Suggested ID columns (safe to remove):** {', '.join(suggested_id_cols)}")

        # Allow user to select any columns to drop
        # Exclude the current target column from the selection options
        column_options = [col for col in df.columns if col != target_col]
        
        columns_to_drop = st.multiselect(
            "Select columns to drop:",
            options=column_options,
            default=suggested_id_cols # Pre-select suggested ID columns for transparency
        )
        
        if columns_to_drop:
            if st.button("Remove Selected Columns"):
                # Check if any selected column is the target column before dropping (robustness)
                if target_col in columns_to_drop:
                     st.error(f"❌ Cannot remove the target column: **{target_col}**.")
                else:
                    df = df.drop(columns=columns_to_drop)
                    st.session_state.df_clean = df
                    st.success(f"✅ Removed {len(columns_to_drop)} columns: {', '.join(columns_to_drop)}")
                    st.rerun()
        else:
            st.info("No columns selected for removal.")
    # END OF MODIFIED SECTION
    
    # MODIFIED SECTION: Quick clean all button (Removing its internal cleaning logic)
    st.subheader("⚡ Quick Actions (Simplified)")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Auto-Clean Target & Duplicates", help="Automatically: normalize target, drop rows with missing target, and remove duplicate rows."):
            try:
                with st.spinner("Auto-cleaning..."):
                    
                    # 1. Normalize Target
                    df_clean = normalize_target(df, target_col)
                    
                    # 2. Drop missing target rows
                    df_clean = df_clean.dropna(subset=[target_col])
                    
                    # 3. Drop Duplicates
                    before_dup = len(df_clean)
                    df_clean = df_clean.drop_duplicates()
                    
                    st.session_state.df_clean = df_clean
                    
                st.success(f"✅ Quick Auto-cleaning complete! ({before_dup - len(df_clean)} duplicates removed)")
                st.rerun()
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("↩️ Reset to Original Data"):
            st.session_state.df_clean = None
            # Target is kept as it was set on the upload page
            st.info("🔄 Reset to original raw data")
            st.rerun()
    # END MODIFIED SECTION
    
    # Show current state
    st.subheader("📊 Current Data Preview")
    st.dataframe(df.head(10), width='stretch')
    
    # Show target distribution if available
    if target_col in df.columns:
        st.subheader("🎯 Target Distribution")
        try:
            # Must normalize target column first if it hasn't been done for correct 0/1 counts
            # NOTE: This is redundant if the target was normalized at the start of the page, but is a robust check
            df_target_check = normalize_target(df.copy(), target_col)
            churn_counts = df_target_check[target_col].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            with col1:
                labels = [str(v) for v in churn_counts.index]
                fig = px.pie(
                    values=churn_counts.values,
                    names=labels,
                    title=f"{target_col} Distribution",
                    color_discrete_sequence=['#667eea', '#764ba2']
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.write("**Value Counts:**")
                for idx, count in churn_counts.items():
                    st.metric(f"Class {idx}", count)
        except:
            st.write(df[target_col].value_counts())
