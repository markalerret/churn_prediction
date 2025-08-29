#run with this command in terminal
#python -m streamlit run streamlit_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Customer Churn Analysis Dashboard",
    layout="wide",
    initial_sidebar_state='expanded'
)

# Load data (you'll need to add your data loading here)
@st.cache_data
def load_data():
    data = pd.read_csv("data/churn_prediction_data.csv")

    # Data cleaning (Based on analysis)
    # 1. TotalCharges
    data.loc[data['tenure'] == 0, 'TotalCharges'] = '0'
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

    # 2. SeniorCitizen
    data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)

    return data

# Helper functions
@st.cache_data
def calculate_risk_scores(data):
    """recalculate risk scores for dashboard"""
    def calc_score(row):
        score = 0
        #Contract risk
        if row['Contract'] == 'Month-to-month':
            score += 3
        elif row['Contract'] == 'One year':
            score += 1

        #Payment risk
        if row['PaymentMethod'] == 'Electronic check':
            score += 2
        elif row['PaymentMethod'] == 'Mailed check':
            score += 1

        #Service risk
        if row['InternetService'] == 'Fiber optic':
            score += 2
        elif row['InternetService'] == 'DSL':
            score += 1

        return score

    return data.apply(calc_score, axis=1)

@st.cache_data
def get_risk_score_distribution(data):
    """Get risk score distribution for dashboard"""
    risk_scores = calculate_risk_scores(data)
    risk_churn = data.groupby(risk_scores)['Churn'].apply(lambda x: (x == 'Yes').mean())
    risk_counts = risk_scores.value_counts().sort_index()

    return {
        'scores': risk_churn.index.tolist(),
        'churn_rates': risk_churn.values.tolist(),
        'customer_counts': [risk_counts[score] for score in risk_churn.index]
    }

@st.cache_data
def get_churn_by_category(data,column):
    """Get churn rates by category for any column"""
    return data.groupby(column)['Churn'].apply(lambda x: (x == 'Yes').mean())


# Main Navigation
def main():
    st.title("Customer Churn Analysis Dashboard")

    # Load data
    data = load_data()
    if data is None:
        st.stop() #Stop execution if data doesn't load

    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "Executive Summary": show_executive_summary,
        "Data Overview": show_data_overview,
        "Key Churn Drivers": show_key_drivers,
        "Fiber Optic Deep Dive": show_fiber_analysis,
        "Risk Score Analysis": show_risk_analysis,
        "Business Recommendations": show_recommendations
    }

    selected_page = st.sidebar.radio("Select Page", list(pages.keys()))

    #Display selected page
    pages[selected_page]()

#Page functions
def show_executive_summary():
    st.header("Executive Summary")

    # Load and test the data
    data = load_data()

    # Display basic data info to confirm it's working
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(data):,}")

    with col2:
        churn_rate = (data['Churn'] == 'Yes').mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")

    with col3:
        avg_tenure = data['tenure'].mean()
        st.metric("Avg Tenure (Months)", f"{avg_tenure:.1f}")

    with col4:
        avg_monthly = data['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_monthly:.0f}")

    st.markdown("---")

    st.markdown("""
    ## Customer Churn Analysis - Key Findings

    **Dataset**: 7,043 customers, 27% churn rate

    **Top Churn Drivers**:
    1. **Contract Type**: Month-to-month customers churn at 42.7% vs 2.8% for two-year contracts
    2. **Payment Method**: Electronic check customers churn at 45.3% vs 15.2% for automatic credit card payments and 16.7% for automatic bank transfer payments
    3. **Internet Service**: Fiber optic customers churn at 41.9% despite premium pricing

    **Major Business Insight**: Fiber optic customers represent a "perfect storm" of risk factors
    - 69% have month-to-month contracts
    - 52% pay by electronic check
    - 89.7% pay premium prices ($70-110/month)
    - **Result**: Compound risk leading to high churn

    **Risk Model**: Risk scores 0-7 show exponential churn increase (1% to 60%)
    """)

def show_data_overview():
    st.header("Data Overview")
    
    data=load_data()

    # Dataset characteristics section
    st.subheader("Dataset Characteristics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(data):,}")

    with col2:
        st.metric("Total Features", len(data.columns))

    with col3:
        churn_rate = (data['Churn'] == 'Yes').mean()
        st.metric("Overall Churn Rate", f"{churn_rate:.1%}")

    with col4:
        avg_tenure = data['tenure'].mean()
        st.metric("Average Tenure (Months)", f"{avg_tenure:.1f}")

    st.markdown("---")

    # Data types breakdown
    st.subheader("Data Types & Structure")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Categorical Variables (16)**:")
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('customerID') #remove ID column
        categorical_cols.remove('Churn') #remove churn column
        for col in categorical_cols:
            unique_count = data[col].nunique()
            st.write(f"{col}: {unique_count} unique values")

        st.write("")

        #Add target variable separately
        st.write("**Target Variable**")
        st.write(f"Churn: {data['Churn'].nunique()} classes (Yes/No)")

    with col2:
        st.write("**Numerical Variables (3)**:")
        numerical_cols = data[['tenure', 'MonthlyCharges', 'TotalCharges']].describe()
        st.dataframe(numerical_cols.round(2))

    st.markdown("---")

    #Data quality section
    st.subheader("Data Quality Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.success("**Quality Checks Passed**:")
        st.write("- No missing values after cleaning")
        st.write("- No duplicate records")
        st.write("- Business Logic Validated")
        st.write("- Data Types Corrected")

    with col2:
        st.info("**Data Cleaning Applied**:")
        st.write("- **TotalCharges Issue**: 11 customers with 0 tenure had empty TotalCharges")
        st.write("- **Root Cause**: New customers haven't accumulated charges")
        st.write("- **Solution**: Set TotalCharges = 0 for tenure = 0 customers")
        st.write("- **Validation**: Confirmed business logic consistency")

    st.markdown("---")

    # Sample data preview
    st.subheader("Sample Data Preview")
    st.write("First 10 rows of the cleaned dataset:")
    st.dataframe(data.head(10))


def show_key_drivers():
    st.header("Key Churn Drivers")
    
    data = load_data()

    st.markdown("""
    Analysis of categorical variables reveals three primary drivers of customer churn,
    each representing different aspects of customer behavior and service preferences.
    """)

    #Create headers above the columns
    header_col1, header_col2, header_col3 = st.columns(3)
    with header_col1:
        st.markdown("<h4 style-'text-align=center; margin-bottom: 10px;'>Contract Type</h4>", unsafe_allow_html=True)
    with header_col2:
        st.markdown("<h4 style-'text-align=center; margin-bottom: 10px;'>Payment Method</h4>", unsafe_allow_html=True)
    with header_col3:
        st.markdown("<h4 style-'text-align=center; margin-bottom: 10px;'>Internet Service</h4>", unsafe_allow_html=True)

    # Create three columns for the charts
    col1, col2, col3 = st.columns(3)

    with col1:
        contract_churn = get_churn_by_category(data, "Contract")

        fig1 = px.bar(
            x = contract_churn.index.tolist(),
            y = contract_churn.values.tolist(),
            labels = {'x': 'Contract Type', 'y':'Churn Rate'},
            color = contract_churn.values,
            color_continuous_scale='Reds'
        )
        fig1.update_layout(
            showlegend=False, 
            height=400,
            margin=dict(l=60, r=60, t=40, b=120),
            xaxis=dict(
                title = "",
                tickangle=45,
                tickfont=dict(size=10),
                automargin=False
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                automargin=False
            ),
        )
        # Format y-axis as percentage
        fig1.update_yaxes(tickformat='.0%')

        st.plotly_chart(fig1, use_container_width=True)

        st.write("**Churn Rates**:")
        for contract, rate in contract_churn.items():
            st.write(f"{contract}: {rate:.1%}")

    with col2:
        payment_churn = get_churn_by_category(data, 'PaymentMethod')

        fig2 = px.bar(
            x = payment_churn.index.tolist(),
            y = payment_churn.values.tolist(),
            labels = {'x': 'Payment Method', 'y': 'Churn Rate'},
            color = payment_churn.values,
            color_continuous_scale = 'Blues'
        )
        fig2.update_layout(
            showlegend=False, 
            height=400,
            margin=dict(l=60, r=60, t=25, b=120),
            xaxis=dict(
                title="",
                tickangle=45,
                tickfont=dict(size=10),
                automargin=False
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                automargin=False
            ),
        )
        # Format y-axis as percentage
        fig2.update_yaxes(tickformat='.0%')

        st.plotly_chart(fig2, use_container_width=True)

        st.write("**Churn Rates**:")
        for payment, rate in payment_churn.items():
            st.write(f"{payment}: {rate:.1%}")

    with col3:
        internet_churn = get_churn_by_category(data, 'InternetService')

        fig3 = px.bar(
            x = internet_churn.index.tolist(),
            y = internet_churn.values.tolist(),
            labels = {'x': 'Internet Service', 'y': 'Churn Rate'},
            color = internet_churn.values,
            color_continuous_scale = 'Greens'
        )
        fig3.update_layout(
            showlegend=False, 
            height=400,
            margin=dict(l=60, r=60, t=40, b=120),
            xaxis=dict(
                title="",
                tickangle=45,
                tickfont=dict(size=10),
                automargin=False
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                automargin=False
            ),
        )
        # Format y-axis as percentage
        fig3.update_yaxes(tickformat='.0%')

        st.plotly_chart(fig3, use_container_width=True)

        st.write("**Churn Rates**:")
        for service, rate in internet_churn.items():
            st.write(f"{service}: {rate:.1%}")

    # Add summart insights
    st.markdown("---")
    st.subheader("Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Contract Commitment Effect**
                
        Month-to-month customers show 15x higher churn that two-year customers, highlighting the importance of customer commitment.        
        """)

    with col2:
        st.info("""
        **Payment Method Signal**
                
        Automatic payment methods correlate with 3x better retention, suggesting customer engagement level affects churn risk.
        """)

    with col3:
        st.info("""
        **Service Paradox**
                
        Premium fiber service shows highest churn despite higher pricing, indicating potential value perception issues.
        """)


def show_fiber_analysis():
    st.header("Fiber Optic Deep Dive")
    
    data = load_data()

    # Introduction
    st.markdown("""
    **The 'Perfect Storm' Discovery**
                
    Fiber Optic customers don't just have high churn, they stack multiple risk factors that compound into a business problem.
    """)

    st.markdown("---")

    # Key metrics section
    st.subheader("Fiber Customer Risk Profile")

    fiber_data = data[data['InternetService'] == 'Fiber optic']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Fiber Customers", f"{len(fiber_data):,}")

    with col2:
        fiber_churn = (fiber_data['Churn'] == 'Yes').mean()
        st.metric("Fiber Churn Rate", f"{fiber_churn:.1%}")

    with col3:
        month_to_month_pct = (fiber_data['Contract'] == 'Month-to-month').mean()
        st.metric("Month-to-Month", f"{month_to_month_pct:.1%}")

    with col4:
        electronic_check_pct = (fiber_data['PaymentMethod'] == 'Electronic check').mean()
        st.metric("Electronic Check", f"{electronic_check_pct:.1%}")

    st.markdown("---")

    # Comparison visualization
    st.subheader("Fiber vs Non-fiber Customer Comparison")

    # Create comparison data
    comparison_metrics = {
        'Customer Type': ['Fiber Optic', 'Non-Fiber'],
        'Churn Rate': [
            (data[data['InternetService'] == 'Fiber optic']['Churn'] == 'Yes').mean(),
            (data[data['InternetService'] != 'Fiber optic']['Churn'] == 'Yes').mean()
        ],
        'Month-to-Month %': [
            (data[data['InternetService'] == 'Fiber optic']['Contract'] == 'Month-to-month').mean(),
            (data[data['InternetService'] != 'Fiber optic']['Contract'] == 'Month-to-month').mean()
        ],
        'Electronic Check %': [
            (data[data['InternetService'] == 'Fiber optic']['PaymentMethod'] == 'Electronic check').mean(),
            (data[data['InternetService'] != 'Fiber optic']['PaymentMethod'] == 'Electronic check').mean()
        ]
    }

    # Create comparison chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Fiber Optic',
        x=['Churn Rate', 'Month-to-Month', 'Electronic Check'],
        y=[comparison_metrics['Churn Rate'][0],
           comparison_metrics['Month-to-Month %'][0],
           comparison_metrics['Electronic Check %'][0]],
        marker_color='lightcoral'
    ))

    fig.add_trace(go.Bar(
        name='Non-Fiber',
        x=['Churn Rate', 'Month-to-Month', 'Electronic Check'],
        y=[comparison_metrics['Churn Rate'][1],
           comparison_metrics['Month-to-Month %'][1],
           comparison_metrics['Electronic Check %'][1]],
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Fiber vs Non-Fiber Customer Risk Factors',
        barmode='group',
        yaxis_title='Percentage',
        height=400
    )

    # Format y-axis as percentage
    fig.update_yaxes(tickformat='.0%')

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Key insights
    st.subheader("Root Cause Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.error("""
        **The Problem**:
                 
        - **Customer Acquisition Issue**: Fiber sevice attracts price-sensitive, commitment-averse customers
                 
        - **Compound Risk Factors**: Fiber customers stack multiple high-risk behaviors
        
        - **Premium Service Paradox**: Highest-priced service has highest churn
        """)

    with col2:
        st.success(r"""
        **Why This Matters**:
                   
        - **Explains Other Patterns**: Electronic check and $70-110 churn driven by fiber overlap
                   
        - **Business Impact**: 89.7% of fiber customers pay $70-110/month
                   
        - **Strategic Insight**: Problem is customer profile, not service quality
        """)

    st.markdown("---")

    # Business Recommendations
    st.subheader("Strategic Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Fix Acquisition**
                
        Target contract-willing customers for fiber service
        """)

    with col2:
        st.info("""
        **Payment Incentives**
                
        Push Automatic payments for fiber customers
        """)

    with col3:
        st.info("""
        **Contract Incentives**
                
        Offer discounts for fiber + contract combinations
        """)


def show_risk_analysis():
    st.header("Risk Score Analysis")
    
    data = load_data()

    # Introduction
    st.markdown("""
    **Quantifying Compound Risk**

    Created a composite risk score (0-7) to demonstrate how multiple churn factors compound together, not just add up individually.
    """)

    st.markdown("---")

    # Risk scoring methodology
    st.subheader("Risk Scoring Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Contract Risk (0-3 points)**
                
        - Two year: 0 points
        - One year: 1 point
        - Month-to-month: 3 points
        """)

    with col2:
        st.info("""
        **Payment Risk (0-2 points)**
                
        - Automatic payments: 0 points
        - Mailed check: 1 point
        - Electronic check: 2 points
        """)

    with col3:
        st.info("""
        **Service Risk (0-2 points)**
                
        - No internet: 0 points
        - DSL: 1 point
        - Fiber optic: 2 points
        """)

    st.markdown("---")

    # Calculate risk scores
    risk_scores = calculate_risk_scores(data)
    risk_data = get_risk_score_distribution(data)

    # Risk score distribution
    st.subheader("Risk Score Distribution & Churn Rates")

    col1, col2 = st.columns([2,1])

    with col1:
        # Create risk score visualization
        fig = px.bar(
            x=risk_data['scores'],
            y=risk_data['churn_rates'],
            title="Churn Rate by Risk Score",
            labels={'x': 'Risk Score', 'y': 'Churn Rate'},
            color=risk_data['churn_rates'],
            color_continuous_scale='Reds'
        )

        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Risk Score",
            yaxis_title="Churn Rate"
        )

        # Add customer count annotations
        for i, (score, rate, count) in enumerate(zip(risk_data['scores'], risk_data['churn_rates'], risk_data['customer_counts'])):
            fig.add_annotation(
                x=score,
                y=rate + 0.02,
                text=f"{rate:.1%}",
                showarrow=False,
                font=dict(size=10, color='black')
            )

            fig.add_annotation(
                x=score,
                y=0.05,
                text=f"n={count}",
                showarrow=False,
                font=dict(size=8, color='gray')
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Risk Score Breakdown**:")
        st.write("")

        for score, rate, count in zip(risk_data['scores'], risk_data['churn_rates'], risk_data['customer_counts']):
            st.write(f"Score {score}: {rate:.1%} churn")
            st.write(f"({count:,} customers)")
            st.write("")

    st.markdown("---")

    # Key insights
    st.subheader("Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Exponential Risk Pattern**
                   
        - **Low Risk (0-2)**: 1-6% churn rates
        - **Medium Risk (3-5)**: 13-36% churn rates
        - **High Risk (6-7)**: 44-60% churn rates
                   
        **Risk Compounds exponentially, not linearly**
        """)

    with col2:
        lowest_churn_pct = 0.8
        highest_churn_pct = 60.4
        risk_multiplier = highest_churn_pct/lowest_churn_pct

        st.error(f"""
        **Extreme Risk Customers**
                 
        - **Highest Risk Customers** have {risk_multiplier:.1f}x higher churn rates than lowest risk
        - **Score 7 customers**: {highest_churn_pct:.1f}% churn rate
        - **Score 0 customers**: {lowest_churn_pct:.1f}% churn rate
                 
        **Immediate intervention needed for high-risk scores**
        """)

    st.markdown("---")

    # Business applications
    st.subheader("Business Applications")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Targeted Retention**
                
        - **Score 6-7**: Immediate intervention with personal calls, retention offers, and account manager assignment
        - **Score 4-5**: Proactive outreach through email campaigns, service reviews, and upgrade incentives
        - **Score 0-3**: Standard loyalty programs with rewards points, referral bonuses, and upsell opportunities
        """)

    with col2:
        st.info("""
        **Customer Segmentation**
                
        - **Risk based pricing**: Offer 10-20% discounts to high-risk customers to improve retention
        - **Customized service levels**: High-risk customers get priority support and dedicated account managers
        - **Differentiated communication**: Tailor messaging frequency and channel based on risk profile
        """)

    with col3:
        st.info("""
        **Predictive Modeling**
                
        - **Feature engineering foundation**: Risk score can be used as input variable in machine learning models
        - **Model input variable**: Compare ML predictions against risk score to ensure business logic alignment
        - **Business rule validation**: Deploy risk score calculation for real-time customer assessment
        """)

    # Risk score implications
    st.markdown("---")

    high_risk_count = sum([count for score, count in zip(risk_data['scores'], risk_data['customer_counts']) if score>= 6])
    total_customers = sum(risk_data['customer_counts'])

    st.warning(f"""
    **Strategic Insight**: {high_risk_count:,} customers ({high_risk_count/total_customers:.1%} of total) have risk scores of 6 or higher, representing the highest-priority segment for immediate retention efforts.
    """)


def show_recommendations():
    st.header("Business Recommendations")
    
    data = load_data()

    # Executive summary of the problem
    st.markdown("""
    **Strategic Action Plan**
                
    Based on conprehensive churn analysis, here are prioritized recommendation to reduce customer churn and improve retention across key risk segments.
    """)

    st.markdown("---")

    # Priority recommendations with impact estimates
    st.subheader("Priority Recommendations")

    # Priority 1
    st.markdown("### 1. Fix Fiber Optic Customer Acquisition Strategy")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        **Problem**: Fiber customers show 41.9% churn rate due to compound risk factors
                 
        **Root Cause**: Attracting commitment-averse customers to premium service
                 
        **Solutions**:
        - Offer 20-30% contract discounts for fiber customers who sign 1+ year agreements
        - Provide additional 5-10% discounts for automatic payment enrollment
        - Restructure marketing to target stability-seeking customers, not price-sensitive ones
        - Create "Fiber + Commitment" bundles with enhanced value proposition
        """)

    with col2:
        fiber_customers = len(data[data['InternetService'] == 'Fiber optic'])
        potential_saves = int(fiber_customers * 0.17) #17% improvement estimate

        st.success(f"""
        **Expected Impact**
                   
        - **Customers at risk**: {fiber_customers:,}
        - **If we acheive 17% improvement**: {potential_saves:,} customers retained
        - **Target**: Reduce 41.9% churn to ~25%
        - **Timeline**: 3-6 months
        """)

    st.markdown("---")

    # Priority 2
    st.markdown("### 2. Implement Risk-based Retention Program")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        **Problem**: Reactive approach to churn - customers leave before intervention
                 
        **Solution**: Deploy risk scoring model for proactive retention
                 
        **Implementation**:
        - **Score 6-7**: immediate personal outreach with retention specialists
        - **Score 4-5**: Automated email campaings with service optimization offers
        - **Score 0-3**: Standard loyalty programs and upselling opportunities
        - **Monthly scoring**: Recalculate risk scores to catch changes in customer behavior
        """)

    with col2:
        high_risk_customers = len(data[calculate_risk_scores(data) >= 6])
        medium_risk_customers = len(data[(calculate_risk_scores(data) >= 4) & (calculate_risk_scores(data) < 6)])

        st.success(f"""
        **Expected Impact**

        - **High-risk customers**: {high_risk_customers:,}
        - **Medium-risk customers**: {medium_risk_customers:,}
        - **If program achieves 15% improvement in high-risk customers**: {int(high_risk_customers * 0.15)} customers retained
        - **Timeline**: 1-2 months
        """)

    st.markdown("---")

    # Priority 3
    st.markdown("### 3. Payment Method Migration Campaign")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        **Problem**: Electronic check customers have 45.3% churn vs 15.2% for automatic payments
                 
        **Solution**: Incentivize migration to automatic payment methods
                 
        **Implementation**:
        - **Financial Incentives**: $5-10/month discount for automatic payment conversion
        - **Education campaign**: Highlight convenience and reliability benefits
        - **Targeted outreach**: Focus on high-risk customers using electronic check
        - **Gradual rollout**: Start with new customers, then target existing customers
        """)

    with col2:
        electronic_check_customers = len(data[data['PaymentMethod'] == 'Electronic check'])
        conversion_potential = int(electronic_check_customers * 0.3) # 30% conversion estimate

        st.success(f"""
        **Expected Impact**
                   
        - **Target customers**: {electronic_check_customers:,}
        - **If 30% convert**: {conversion_potential:,} customers retained
        - **Timeline**: 2-4 months
        """)

    st.markdown("---")

    # Implementation roadmap
    st.subheader("Implementation Roadmap")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Phase 1 (Month 1-2)**
                
        - Deploy risk scoring system
        - Launch retention program
        - Begin payment method campaign
        """)

    with col2:
        st.info("""
        **Phase 2 (Month 3-4)**
                
        - Implement fiber acquisition changes
        - Analyze early results
        - Refine targeting strategies
        """)

    with col3:
        st.info("""
        **Phase 3 (Month 5-6)**
                
        - Scale successful programs
        - Measure full impact
        - Plan next optimization cycle
        """)

    st.markdown("---")

    # Success metrics
    st.subheader("Success Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **Primary KPIs**:
        - **Overall churn rate**: reduction from 26.5% to <20%
        - **Fiber optic churn**: reduction from 41.9% to ~25%
        - **High-risk customer retention**: improvement by 15%
        - **Payment method conversion**: rate of 25-30%
        """)

    with col2:
        st.write("""
        **Secondary Metrics**:
        - **Customer lifetime value** increase
        - **Revenue retention** improvement
        - **Customer satisfaction** scores
        - **Cost per acquisition** optimization
        """)

    st.markdown("---")

    # Resource requirements
    st.subheader("Resource Requirements")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.warning("""
        **Technology**
                   
        - Risk scoring system deployment
        - Automated campaign tools
        - Customer segmentation platform
        """)

    with col2:
        st.warning("""
        **Personnel**
                   
        - Retention specialists (2-3 FTE)
        - Campaign managers
        - Data analyst support
        """)

    with col3:
        st.warning("""
        **Budget**
        
        - Customer incentives/discounts
        - Technology implementation
        - Staff training and support
        """)

    st.markdown("---")

    #Final call to action
    current_churn = (data['Churn'] == 'Yes').mean()
    annual_churned = int(len(data) * current_churn)

    st.error(f"""
    **Urgency**: With {annual_churned:,} customers churning annually at the current {current_churn:.1%} rate, implementing these recommendation could save 500-1,000 customers per year and significantly improve revenue retention.
    """)

    st.success("""
    **Next Steps**:
    1. Secure stakeholder buy-in for phased implementation
    2. Allocate resources for Phase 1 deployment
    3. Establish measurement framework for tracking success
    4. Begin immediate deployment of risk scoring system
    """)


if __name__ == "__main__":
    main()