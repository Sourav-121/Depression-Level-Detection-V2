def get_recommendations(depression_level, high_risk_symptoms):
    """
    Get personalized recommendations based on depression level and symptoms
    """
    recommendations = {}
    
    # Base recommendations for all levels
    base_recs = [
        " Maintain a consistent daily routine",
        " Practice mindfulness or meditation for 10-15 minutes daily",
        " Keep a journal to track your thoughts and feelings",
        " Engage in regular physical activity (30 minutes, 5 times a week)",
        " Eat balanced meals at regular times and stay hydrated",
        " Prioritize getting 7-9 hours of quality sleep",
        " Stay connected with supportive friends and family members",
        " Set small, achievable goals each day to build momentum"
    ]
    
    recommendations["🌿 General Self-Care"] = base_recs
    
    # Level-specific recommendations
    if depression_level == "No Depression":
        recommendations[" Maintaining Mental Wellness"] = [
            "Continue practicing good self-care habits you've developed",
            "Schedule regular check-ins with yourself to monitor changes",
            "Develop and nurture a strong support network",
            "Learn stress management techniques proactively",
            "Engage in activities that bring you joy and fulfillment",
            "Consider volunteering or helping others to boost well-being"
        ]
        
    elif depression_level == "Mild Depression":
        recommendations["🟡 Managing Mild Depression"] = [
            "Consider speaking with a counselor or therapist for early intervention",
            "Join a support group to connect with others experiencing similar feelings",
            "Practice cognitive behavioral therapy (CBT) techniques on your own",
            "Limit social media and screen time, especially before bed",
            "Spend time in nature - even 20 minutes outdoors can help",
            "Challenge negative thoughts by writing down alternatives",
            "Establish a consistent sleep schedule, even on weekends",
            "Try the 5-4-3-2-1 grounding technique when feeling overwhelmed"
        ]
        
    elif depression_level == "Moderate Depression":
        recommendations["🟠 Addressing Moderate Depression"] = [
            "Seek professional help immediately - schedule an appointment with a mental health provider",
            "Consider a combination of therapy and medication (consult a psychiatrist)",
            "Reach out to trusted friends or family and share what you're experiencing",
            "Break tasks into smaller, manageable steps to avoid overwhelm",
            "Use a mood tracking app to identify patterns and triggers",
            "Avoid alcohol and drugs as they can worsen depression",
            "Practice self-compassion - treat yourself with the kindness you'd show a friend"
        ]
        
    else:  # Severe Depression
        recommendations[" Urgent Support for Severe Depression"] = [
            " **IMMEDIATE ACTION REQUIRED** - Contact a mental health crisis service NOW"
        ]
    
    # Symptom-specific recommendations
    symptom_recs = []
    
    # Map symptoms to specific recommendations
    symptom_map = {
        'Sleep Disruption': [
            "Maintain a consistent sleep-wake schedule",
            "Avoid screens 1 hour before bedtime",
            "Create a relaxing bedtime routine (warm bath, reading)",
            "Limit caffeine after 2 PM"
        ],
        'Sleepdisruption': [  # Alternative format
            "Maintain a consistent sleep-wake schedule",
            "Avoid screens 1 hour before bedtime",
            "Create a relaxing bedtime routine (warm bath, reading)",
            "Limit caffeine after 2 PM"
        ],
        'Loss Of Appetite': [
            "Eat small, frequent meals throughout the day",
            "Choose nutrient-dense foods even if portions are small",
            "Set reminders to eat at regular times",
            "Consider meal prepping when you have more energy"
        ],
        'Self Harm': [
            " This is a medical emergency - seek help immediately",
            "Call a crisis hotline right now: 988",
            "Remove any means of self-harm from your vicinity",
            "Do not stay alone - contact someone you trust"
        ],
        'Hopelessness': [
            "Challenge hopeless thoughts by looking for small evidence to the contrary",
            "Remember past difficulties you've overcome",
            "Focus on what you can control in the present moment",
            "Talk to someone who can provide perspective"
        ],
        'Loneliness': [
            "Join a club or group related to your interests",
            "Schedule regular calls with friends or family",
            "Consider volunteering to connect with others",
            "Try online communities focused on mental health support"
        ],
        'Chronic Fatigue': [
            "Break tasks into smaller, manageable chunks",
            "Schedule rest periods throughout the day",
            "Gentle exercise like walking can actually increase energy",
            "Check vitamin levels (especially B12, D, iron) with your doctor"
        ],
        'Anxiety': [
            "Practice deep breathing exercises (4-7-8 technique)",
            "Use grounding techniques when feeling overwhelmed",
            "Limit caffeine and sugar intake",
            "Try progressive muscle relaxation before bed"
        ]
    }
    
    for symptom in high_risk_symptoms:
        for key, recs in symptom_map.items():
            if key.lower() in symptom.lower():
                symptom_recs.extend(recs)
    
    if symptom_recs:
        # Remove duplicates while preserving order
        unique_recs = []
        for rec in symptom_recs:
            if rec not in unique_recs:
                unique_recs.append(rec)
        recommendations[" Targeted Support for Your Symptoms"] = unique_recs[:5]  # Limit to 5
    
    return recommendations