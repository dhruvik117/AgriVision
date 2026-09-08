
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AgriVision | Crop Disease AI",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .brand {
        text-align: center;
        padding: 10px 0 5px 0;
    }

    .brand h1 {
        font-size: 2.7rem;
        margin-bottom: 0;
        font-weight: 800;
    }

    .brand p {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-top: 5px;
    }

    .result-box {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin: 15px 0;
    }

    .info-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.22);
        margin: 10px 0;
    }

    .score {
        font-size: 1.35rem;
        font-weight: 700;
    }

    .small-note {
        font-size: 0.88rem;
        opacity: 0.70;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL + CLASS NAMES
# ============================================================

MODEL_PATH = "model/AgriVision_V2.keras"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

# ============================================================
# MULTILINGUAL UI TEXT
# ============================================================

UI = {
    "English": {
        "subtitle": "AI-powered crop health screening for faster plant disease identification",
        "language": "Language",
        "upload": "Upload a clear image of a crop leaf",
        "upload_help": "For best results, photograph one leaf clearly in good lighting.",
        "analyze": "🔍 Analyze Leaf",
        "analyzing": "Analyzing the leaf...",
        "result": "AI Screening Result",
        "crop": "Crop",
        "condition": "Detected Condition",
        "score": "Model Score",
        "higher": "🟢 Higher model confidence",
        "moderate": "🟡 Moderate model confidence",
        "verify": "🟠 Verification recommended",
        "symptoms": "🔍 Symptoms",
        "causes": "🦠 Cause",
        "treatment": "💊 Treatment / Management",
        "prevention": "🛡️ Prevention",
        "alternatives": "View alternative predictions",
        "technical": "Technical model label",
        "warning": "⚠️ Important",
        "safety": "AgriVision is a screening and educational tool. It does not replace professional agricultural diagnosis.",
        "wrong_warning": "AI predictions can be incorrect, especially with poor lighting, unusual symptoms, or images outside the training data.",
        "about": "About AgriVision",
        "about_text": "AgriVision is an AI-based crop health screening prototype designed to help identify common diseases in bell pepper, potato and tomato leaves.",
        "supported": "Supported crops: Bell Pepper • Potato • Tomato",
        "model": "Model: MobileNetV2 transfer learning",
        "classes": "Disease classes: 15",
        "next_step": "Recommended next step",
        "healthy_note": "No major disease pattern was detected. Continue normal crop monitoring and good agricultural practices.",
        "not_available": "Disease information is not available for this class."
    },

    "Hindi": {
        "subtitle": "तेज़ पौध रोग पहचान के लिए AI आधारित फसल स्वास्थ्य जाँच",
        "language": "भाषा",
        "upload": "फसल की पत्ती की स्पष्ट तस्वीर अपलोड करें",
        "upload_help": "बेहतर परिणाम के लिए एक पत्ती की साफ तस्वीर अच्छी रोशनी में लें।",
        "analyze": "🔍 पत्ती की जाँच करें",
        "analyzing": "पत्ती की जाँच की जा रही है...",
        "result": "AI जाँच का परिणाम",
        "crop": "फसल",
        "condition": "पहचानी गई स्थिति",
        "score": "मॉडल स्कोर",
        "higher": "🟢 मॉडल का विश्वास अपेक्षाकृत अधिक",
        "moderate": "🟡 मॉडल का विश्वास मध्यम",
        "verify": "🟠 पुष्टि की सलाह दी जाती है",
        "symptoms": "🔍 लक्षण",
        "causes": "🦠 कारण",
        "treatment": "💊 उपचार / प्रबंधन",
        "prevention": "🛡️ बचाव",
        "alternatives": "अन्य संभावित परिणाम देखें",
        "technical": "तकनीकी मॉडल नाम",
        "warning": "⚠️ महत्वपूर्ण",
        "safety": "AgriVision एक प्रारंभिक जाँच और शैक्षिक उपकरण है। यह कृषि विशेषज्ञ की पेशेवर जाँच का विकल्प नहीं है।",
        "wrong_warning": "AI का परिणाम गलत हो सकता है, विशेषकर खराब रोशनी, असामान्य लक्षण या प्रशिक्षण डेटा से अलग तस्वीरों में।",
        "about": "AgriVision के बारे में",
        "about_text": "AgriVision एक AI आधारित फसल स्वास्थ्य जाँच प्रोटोटाइप है, जो शिमला मिर्च, आलू और टमाटर की पत्तियों में सामान्य रोगों की पहचान में सहायता करता है।",
        "supported": "समर्थित फसलें: शिमला मिर्च • आलू • टमाटर",
        "model": "मॉडल: MobileNetV2 ट्रांसफर लर्निंग",
        "classes": "रोग वर्ग: 15",
        "next_step": "अगला सुझाया गया कदम",
        "healthy_note": "किसी प्रमुख रोग का पैटर्न नहीं मिला। सामान्य फसल निगरानी और अच्छी कृषि पद्धतियाँ जारी रखें।",
        "not_available": "इस वर्ग के लिए रोग संबंधी जानकारी उपलब्ध नहीं है।"
    },

    "Gujarati": {
        "subtitle": "ઝડપી છોડના રોગની ઓળખ માટે AI આધારિત પાક આરોગ્ય તપાસ",
        "language": "ભાષા",
        "upload": "પાકના પાનની સ્પષ્ટ તસવીર અપલોડ કરો",
        "upload_help": "સારા પરિણામ માટે એક પાનની સ્પષ્ટ તસવીર સારી રોશનીમાં લો.",
        "analyze": "🔍 પાનની તપાસ કરો",
        "analyzing": "પાનની તપાસ કરવામાં આવી રહી છે...",
        "result": "AI તપાસનું પરિણામ",
        "crop": "પાક",
        "condition": "ઓળખાયેલી સ્થિતિ",
        "score": "મોડેલ સ્કોર",
        "higher": "🟢 મોડેલનો વિશ્વાસ પ્રમાણમાં વધુ",
        "moderate": "🟡 મોડેલનો વિશ્વાસ મધ્યમ",
        "verify": "🟠 ચકાસણીની સલાહ",
        "symptoms": "🔍 લક્ષણો",
        "causes": "🦠 કારણ",
        "treatment": "💊 સારવાર / વ્યવસ્થાપન",
        "prevention": "🛡️ બચાવ",
        "alternatives": "અન્ય સંભવિત પરિણામો જુઓ",
        "technical": "ટેકનિકલ મોડેલ નામ",
        "warning": "⚠️ મહત્વપૂર્ણ",
        "safety": "AgriVision એક પ્રાથમિક તપાસ અને શૈક્ષણિક સાધન છે. તે કૃષિ નિષ્ણાતની વ્યાવસાયિક તપાસનું સ્થાન લેતું નથી.",
        "wrong_warning": "AIનું પરિણામ ખોટું હોઈ શકે છે, ખાસ કરીને ખરાબ પ્રકાશ, અસામાન્ય લક્ષણો અથવા તાલીમ ડેટાથી અલગ તસવીરોમાં.",
        "about": "AgriVision વિશે",
        "about_text": "AgriVision એક AI આધારિત પાક આરોગ્ય તપાસ પ્રોટોટાઇપ છે, જે શિમલા મરચાં, બટાકા અને ટામેટાના પાનમાં સામાન્ય રોગોની ઓળખ કરવામાં મદદ કરે છે.",
        "supported": "સમર્થિત પાક: શિમલા મરચાં • બટાકા • ટામેટા",
        "model": "મોડેલ: MobileNetV2 ટ્રાન્સફર લર્નિંગ",
        "classes": "રોગ વર્ગો: 15",
        "next_step": "આગળનું સૂચિત પગલું",
        "healthy_note": "કોઈ મુખ્ય રોગનું પેટર્ન મળ્યું નથી. સામાન્ય પાકની દેખરેખ અને સારી ખેતી પદ્ધતિઓ ચાલુ રાખો.",
        "not_available": "આ વર્ગ માટે રોગની માહિતી ઉપલબ્ધ નથી."
    }
}

# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

"Pepper__bell___Bacterial_spot": {
"English": {
"disease": "Pepper Bacterial Spot",
"crop": "Bell Pepper",
"symptoms": "Small dark, brown or water-soaked spots may appear on leaves. Severe infection can cause yellowing and leaf drop.",
"cause": "Usually caused by Xanthomonas bacteria. The bacteria can spread through infected plant material, rain splash and contaminated tools.",
"treatment": "Remove severely affected plant material where practical. Avoid working with wet plants, improve airflow and use clean planting material. For chemical control, follow locally approved agricultural recommendations.",
"prevention": "Use healthy seeds or seedlings, avoid prolonged leaf wetness, provide good spacing and sanitize tools."
},
"Hindi": {
"disease": "शिमला मिर्च का बैक्टीरियल स्पॉट",
"crop": "शिमला मिर्च",
"symptoms": "पत्तियों पर छोटे गहरे, भूरे या पानी जैसे धब्बे दिखाई दे सकते हैं। गंभीर संक्रमण में पत्तियाँ पीली होकर गिर सकती हैं।",
"cause": "यह सामान्यतः Xanthomonas बैक्टीरिया से होता है। संक्रमित पौध सामग्री, बारिश की छींटों और दूषित औजारों से फैल सकता है।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित पौध भागों को हटा दें। गीले पौधों को छूने से बचें, हवा का अच्छा प्रवाह रखें और साफ रोपण सामग्री इस्तेमाल करें। रासायनिक नियंत्रण के लिए स्थानीय रूप से स्वीकृत कृषि सलाह का पालन करें।",
"prevention": "स्वस्थ बीज या पौध लें, पत्तियों को लंबे समय तक गीला न रहने दें, उचित दूरी रखें और औजार साफ रखें।"
},
"Gujarati": {
"disease": "શિમલા મરચાંનો બેક્ટેરિયલ સ્પોટ",
"crop": "શિમલા મરચાં",
"symptoms": "પાન પર નાના ઘેરા, ભૂરા અથવા પાણી જેવા ડાઘ દેખાઈ શકે છે. ગંભીર ચેપમાં પાન પીળા થઈને પડી શકે છે.",
"cause": "આ રોગ સામાન્ય રીતે Xanthomonas બેક્ટેરિયાથી થાય છે. ચેપગ્રસ્ત છોડની સામગ્રી, વરસાદના છાંટા અને દૂષિત સાધનો દ્વારા ફેલાઈ શકે છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત છોડના ભાગો દૂર કરો. ભીના છોડને સ્પર્શ કરવાનું ટાળો, હવાનો સારો પ્રવાહ રાખો અને સ્વચ્છ રોપણી સામગ્રી વાપરો. રાસાયણિક નિયંત્રણ માટે સ્થાનિક રીતે માન્ય કૃષિ સલાહ અનુસરો.",
"prevention": "સ્વસ્થ બીજ અથવા રોપા વાપરો, પાનને લાંબા સમય સુધી ભીના ન રહેવા દો, યોગ્ય અંતર રાખો અને સાધનો સાફ રાખો."
}},

"Pepper__bell___healthy": {
"English": {
"disease": "Healthy Bell Pepper",
"crop": "Bell Pepper",
"symptoms": "No major disease pattern was detected in the uploaded image.",
"cause": "No clear disease pattern detected.",
"treatment": "No disease-specific treatment is indicated by this screening result. Continue normal crop care and monitoring.",
"prevention": "Maintain good spacing, balanced nutrition, proper watering and regular inspection."
},
"Hindi": {
"disease": "स्वस्थ शिमला मिर्च",
"crop": "शिमला मिर्च",
"symptoms": "अपलोड की गई तस्वीर में किसी प्रमुख रोग का पैटर्न नहीं मिला।",
"cause": "कोई स्पष्ट रोग पैटर्न नहीं मिला।",
"treatment": "इस जाँच के आधार पर किसी विशेष रोग उपचार की आवश्यकता नहीं बताई गई है। सामान्य देखभाल और निगरानी जारी रखें।",
"prevention": "उचित दूरी, संतुलित पोषण, सही सिंचाई और नियमित निरीक्षण रखें।"
},
"Gujarati": {
"disease": "સ્વસ્થ શિમલા મરચાં",
"crop": "શિમલા મરચાં",
"symptoms": "અપલોડ કરેલી તસવીરમાં કોઈ મુખ્ય રોગનું પેટર્ન મળ્યું નથી.",
"cause": "કોઈ સ્પષ્ટ રોગનું પેટર્ન મળ્યું નથી.",
"treatment": "આ તપાસના આધારે કોઈ ખાસ રોગની સારવાર સૂચવાતી નથી. સામાન્ય સંભાળ અને દેખરેખ ચાલુ રાખો.",
"prevention": "યોગ્ય અંતર, સંતુલિત પોષણ, યોગ્ય સિંચાઈ અને નિયમિત તપાસ રાખો."
}},

"Potato___Early_blight": {
"English": {
"disease": "Potato Early Blight",
"crop": "Potato",
"symptoms": "Dark brown circular or irregular lesions, often with concentric rings, commonly develop on older leaves.",
"cause": "Usually associated with the fungus Alternaria solani. Disease pressure can increase under plant stress and favorable moisture conditions.",
"treatment": "Remove badly affected leaves when practical and maintain field sanitation. Improve airflow and avoid prolonged leaf wetness. Follow locally approved fungicide guidance if chemical control is needed.",
"prevention": "Use healthy planting material, rotate crops, remove infected debris and maintain good plant nutrition and irrigation."
},
"Hindi": {
"disease": "आलू का अगेती झुलसा रोग",
"crop": "आलू",
"symptoms": "पुरानी पत्तियों पर गहरे भूरे गोल या अनियमित धब्बे बन सकते हैं। इनमें अक्सर गोल-गोल छल्ले दिखाई देते हैं।",
"cause": "यह सामान्यतः Alternaria solani नामक फफूंद से जुड़ा होता है। पौधों पर तनाव और अनुकूल नमी की स्थिति में रोग बढ़ सकता है।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित पत्तियाँ हटा दें और खेत की स्वच्छता रखें। हवा का प्रवाह बढ़ाएँ और पत्तियों को लंबे समय तक गीला न रहने दें। रासायनिक नियंत्रण की आवश्यकता होने पर स्थानीय रूप से स्वीकृत फफूंदनाशी सलाह का पालन करें।",
"prevention": "स्वस्थ रोपण सामग्री लें, फसल चक्र अपनाएँ, संक्रमित अवशेष हटाएँ और उचित पोषण व सिंचाई रखें।"
},
"Gujarati": {
"disease": "બટાકાનો વહેલો ઝાંખપ રોગ",
"crop": "બટાકા",
"symptoms": "જૂના પાન પર ઘેરા ભૂરા ગોળ અથવા અનિયમિત ડાઘ દેખાય છે. તેમાં ઘણી વખત ગોળ ગોળ વલયો દેખાય છે.",
"cause": "આ રોગ સામાન્ય રીતે Alternaria solani નામની ફૂગ સાથે જોડાયેલો છે. છોડ પર તણાવ અને અનુકૂળ ભેજની સ્થિતિમાં રોગ વધી શકે છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત પાન દૂર કરો અને ખેતરની સ્વચ્છતા રાખો. હવાનો પ્રવાહ સુધારો અને પાનને લાંબા સમય સુધી ભીના ન રહેવા દો. રાસાયણિક નિયંત્રણની જરૂર હોય તો સ્થાનિક રીતે માન્ય ફૂગનાશક અંગેની સલાહ અનુસરો.",
"prevention": "સ્વસ્થ રોપણી સામગ્રી વાપરો, પાક ફેરબદલી કરો, ચેપગ્રસ્ત અવશેષો દૂર કરો અને યોગ્ય પોષણ તથા સિંચાઈ રાખો."
}},

"Potato___Late_blight": {
"English": {
"disease": "Potato Late Blight",
"crop": "Potato",
"symptoms": "Irregular dark green, brown or black-looking lesions may appear on leaves. Under favorable conditions the disease can spread rapidly.",
"cause": "Caused by the oomycete Phytophthora infestans. Cool, humid and wet conditions can favor disease development.",
"treatment": "Remove or isolate severely affected plant material where practical and reduce prolonged leaf wetness. Inspect nearby plants because the disease can spread quickly. Use only locally approved control measures under agricultural guidance.",
"prevention": "Use healthy seed tubers, improve airflow, avoid unnecessary leaf wetness and regularly inspect the crop."
},
"Hindi": {
"disease": "आलू का पछेती झुलसा रोग",
"crop": "आलू",
"symptoms": "पत्तियों पर अनियमित गहरे हरे, भूरे या काले जैसे धब्बे दिखाई दे सकते हैं। अनुकूल परिस्थितियों में रोग तेजी से फैल सकता है।",
"cause": "यह Phytophthora infestans नामक ओओमाइसीट से होता है। ठंडी, नम और गीली परिस्थितियाँ रोग के विकास को बढ़ावा दे सकती हैं।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित पौध भागों को हटा दें और पत्तियों के लंबे समय तक गीले रहने को कम करें। आसपास के पौधों की भी जाँच करें। नियंत्रण के लिए केवल स्थानीय रूप से स्वीकृत कृषि उपाय अपनाएँ।",
"prevention": "स्वस्थ बीज कंद लें, हवा का प्रवाह अच्छा रखें, अनावश्यक पत्ती गीलापन कम करें और फसल की नियमित जाँच करें।"
},
"Gujarati": {
"disease": "બટાકાનો મોડો ઝાંખપ રોગ",
"crop": "બટાકા",
"symptoms": "પાન પર અનિયમિત ઘેરા લીલા, ભૂરા અથવા કાળા જેવા ડાઘ દેખાઈ શકે છે. અનુકૂળ પરિસ્થિતિમાં રોગ ઝડપથી ફેલાઈ શકે છે.",
"cause": "આ રોગ Phytophthora infestans નામના ઓઓમાયસિટથી થાય છે. ઠંડી, ભેજવાળી અને ભીની પરિસ્થિતિ રોગના વિકાસને અનુકૂળ હોય છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત છોડના ભાગો દૂર કરો અને પાન લાંબા સમય સુધી ભીના ન રહે તે ધ્યાન રાખો. નજીકના છોડની પણ તપાસ કરો. નિયંત્રણ માટે માત્ર સ્થાનિક રીતે માન્ય કૃષિ પદ્ધતિઓ અપનાવો.",
"prevention": "સ્વસ્થ બીજ કંદ વાપરો, હવાનો સારો પ્રવાહ રાખો, બિનજરૂરી પાન ભીનાશ ટાળો અને પાકની નિયમિત તપાસ કરો."
}},

"Potato___healthy": {
"English": {
"disease": "Healthy Potato",
"crop": "Potato",
"symptoms": "No major disease pattern was detected in the uploaded image.",
"cause": "No clear disease pattern detected.",
"treatment": "No disease-specific treatment is indicated by this screening result. Continue normal crop care and monitoring.",
"prevention": "Use healthy seed tubers, maintain good nutrition and irrigation, and inspect plants regularly."
},
"Hindi": {
"disease": "स्वस्थ आलू",
"crop": "आलू",
"symptoms": "अपलोड की गई तस्वीर में किसी प्रमुख रोग का पैटर्न नहीं मिला।",
"cause": "कोई स्पष्ट रोग पैटर्न नहीं मिला।",
"treatment": "इस जाँच के आधार पर किसी विशेष रोग उपचार की आवश्यकता नहीं बताई गई है। सामान्य देखभाल और निगरानी जारी रखें।",
"prevention": "स्वस्थ बीज कंद लें, उचित पोषण और सिंचाई रखें तथा पौधों का नियमित निरीक्षण करें।"
},
"Gujarati": {
"disease": "સ્વસ્થ બટાકા",
"crop": "બટાકા",
"symptoms": "અપલોડ કરેલી તસવીરમાં કોઈ મુખ્ય રોગનું પેટર્ન મળ્યું નથી.",
"cause": "કોઈ સ્પષ્ટ રોગનું પેટર્ન મળ્યું નથી.",
"treatment": "આ તપાસના આધારે કોઈ ખાસ રોગની સારવાર સૂચવાતી નથી. સામાન્ય સંભાળ અને દેખરેખ ચાલુ રાખો.",
"prevention": "સ્વસ્થ બીજ કંદ વાપરો, યોગ્ય પોષણ અને સિંચાઈ રાખો અને છોડની નિયમિત તપાસ કરો."
}},

"Tomato_Bacterial_spot": {
"English": {
"disease": "Tomato Bacterial Spot",
"crop": "Tomato",
"symptoms": "Small dark spots may develop on leaves. Severe infections can cause yellowing, leaf damage and reduced plant vigor.",
"cause": "Commonly caused by Xanthomonas bacteria and spread through infected material, rain splash, irrigation water and contaminated tools.",
"treatment": "Remove severely affected material where practical, improve airflow and avoid handling wet plants. Use clean planting material and follow locally approved agricultural control recommendations.",
"prevention": "Use healthy seeds or seedlings, avoid overhead watering when possible, maintain spacing and sanitize tools."
},
"Hindi": {
"disease": "टमाटर का बैक्टीरियल स्पॉट",
"crop": "टमाटर",
"symptoms": "पत्तियों पर छोटे गहरे धब्बे बन सकते हैं। गंभीर संक्रमण में पत्तियाँ पीली हो सकती हैं और पौधे की वृद्धि प्रभावित हो सकती है।",
"cause": "यह सामान्यतः Xanthomonas बैक्टीरिया से होता है और संक्रमित सामग्री, बारिश की छींटों, सिंचाई के पानी तथा दूषित औजारों से फैल सकता है।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित भाग हटाएँ, हवा का प्रवाह सुधारें और गीले पौधों को छूने से बचें। साफ रोपण सामग्री का उपयोग करें और स्थानीय कृषि नियंत्रण सलाह का पालन करें।",
"prevention": "स्वस्थ बीज या पौध लें, संभव हो तो ऊपर से पानी देने से बचें, उचित दूरी रखें और औजारों को साफ करें।"
},
"Gujarati": {
"disease": "ટામેટાનો બેક્ટેરિયલ સ્પોટ",
"crop": "ટામેટા",
"symptoms": "પાન પર નાના ઘેરા ડાઘ થઈ શકે છે. ગંભીર ચેપમાં પાન પીળા થઈ શકે છે અને છોડની વૃદ્ધિ ઘટી શકે છે.",
"cause": "આ રોગ સામાન્ય રીતે Xanthomonas બેક્ટેરિયાથી થાય છે અને ચેપગ્રસ્ત સામગ્રી, વરસાદના છાંટા, સિંચાઈના પાણી અને દૂષિત સાધનો દ્વારા ફેલાઈ શકે છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત ભાગો દૂર કરો, હવાનો પ્રવાહ સુધારો અને ભીના છોડને સ્પર્શ કરવાનું ટાળો. સ્વચ્છ રોપણી સામગ્રી વાપરો અને સ્થાનિક કૃષિ નિયંત્રણ સલાહ અનુસરો.",
"prevention": "સ્વસ્થ બીજ અથવા રોપા વાપરો, શક્ય હોય તો ઉપરથી પાણી આપવાનું ટાળો, યોગ્ય અંતર રાખો અને સાધનો સાફ રાખો."
}},

"Tomato_Early_blight": {
"English": {
"disease": "Tomato Early Blight",
"crop": "Tomato",
"symptoms": "Dark lesions with concentric ring patterns may appear, especially on older leaves. Yellowing around lesions can occur.",
"cause": "Usually associated with Alternaria solani. Plant stress and favorable moisture conditions can increase disease development.",
"treatment": "Remove badly affected leaves where practical, improve airflow and maintain sanitation. Avoid prolonged leaf wetness and follow locally approved fungicide guidance when required.",
"prevention": "Use healthy plants, rotate crops, remove infected debris and maintain balanced irrigation and nutrition."
},
"Hindi": {
"disease": "टमाटर का अगेती झुलसा रोग",
"crop": "टमाटर",
"symptoms": "विशेषकर पुरानी पत्तियों पर गहरे धब्बे बन सकते हैं जिनमें गोल-गोल छल्ले दिखाई देते हैं। धब्बों के आसपास पीलापन हो सकता है।",
"cause": "यह सामान्यतः Alternaria solani से जुड़ा होता है। पौधे पर तनाव और अनुकूल नमी की स्थिति रोग बढ़ा सकती है।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित पत्तियाँ हटाएँ, हवा का प्रवाह सुधारें और स्वच्छता रखें। पत्तियों को लंबे समय तक गीला न रहने दें और आवश्यकता पर स्थानीय रूप से स्वीकृत फफूंदनाशी सलाह लें।",
"prevention": "स्वस्थ पौधों का उपयोग करें, फसल चक्र अपनाएँ, संक्रमित अवशेष हटाएँ और संतुलित सिंचाई व पोषण रखें।"
},
"Gujarati": {
"disease": "ટામેટાનો વહેલો ઝાંખપ રોગ",
"crop": "ટામેટા",
"symptoms": "ખાસ કરીને જૂના પાન પર ઘેરા ડાઘ અને તેમાં ગોળ વલય જેવા નિશાન દેખાઈ શકે છે. ડાઘની આસપાસ પીળાશ થઈ શકે છે.",
"cause": "આ રોગ સામાન્ય રીતે Alternaria solani સાથે જોડાયેલો છે. છોડ પર તણાવ અને અનુકૂળ ભેજની સ્થિતિ રોગ વધારી શકે છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત પાન દૂર કરો, હવાનો પ્રવાહ સુધારો અને સ્વચ્છતા રાખો. પાન લાંબા સમય સુધી ભીના ન રહેવા દો અને જરૂર પડે ત્યારે સ્થાનિક રીતે માન્ય ફૂગનાશક અંગેની સલાહ લો.",
"prevention": "સ્વસ્થ છોડ વાપરો, પાક ફેરબદલી કરો, ચેપગ્રસ્ત અવશેષો દૂર કરો અને સંતુલિત સિંચાઈ તથા પોષણ રાખો."
}},

"Tomato_Late_blight": {
"English": {
"disease": "Tomato Late Blight",
"crop": "Tomato",
"symptoms": "Water-soaked or irregular dark lesions can develop on leaves and may expand quickly under cool, wet conditions.",
"cause": "Caused by Phytophthora infestans. Cool and humid weather with prolonged leaf wetness favors the disease.",
"treatment": "Remove severely affected material where practical and reduce prolonged leaf wetness. Check nearby plants because the disease can spread rapidly. Follow locally approved agricultural control advice.",
"prevention": "Improve airflow, avoid unnecessary leaf wetness, inspect plants frequently and use healthy planting material."
},
"Hindi": {
"disease": "टमाटर का पछेती झुलसा रोग",
"crop": "टमाटर",
"symptoms": "पत्तियों पर पानी जैसे या अनियमित गहरे धब्बे बन सकते हैं और ठंडी, गीली परिस्थितियों में तेजी से बढ़ सकते हैं।",
"cause": "यह Phytophthora infestans से होता है। ठंडे और नम मौसम तथा लंबे समय तक पत्तियों के गीले रहने से रोग बढ़ सकता है।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित भाग हटाएँ और पत्तियों के लंबे समय तक गीले रहने को कम करें। आसपास के पौधों की जाँच करें क्योंकि रोग तेजी से फैल सकता है। स्थानीय कृषि नियंत्रण सलाह का पालन करें।",
"prevention": "हवा का प्रवाह सुधारें, अनावश्यक पत्ती गीलापन कम करें, नियमित निरीक्षण करें और स्वस्थ रोपण सामग्री इस्तेमाल करें।"
},
"Gujarati": {
"disease": "ટામેટાનો મોડો ઝાંખપ રોગ",
"crop": "ટામેટા",
"symptoms": "પાન પર પાણી જેવા અથવા અનિયમિત ઘેરા ડાઘ થઈ શકે છે અને ઠંડી, ભીની પરિસ્થિતિમાં ઝડપથી વધી શકે છે.",
"cause": "આ રોગ Phytophthora infestans થી થાય છે. ઠંડુ અને ભેજવાળું હવામાન તથા લાંબા સમય સુધી પાન ભીના રહેવું રોગને અનુકૂળ છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત ભાગો દૂર કરો અને પાન લાંબા સમય સુધી ભીના ન રહે તે ધ્યાન રાખો. નજીકના છોડની તપાસ કરો કારણ કે રોગ ઝડપથી ફેલાઈ શકે છે. સ્થાનિક કૃષિ નિયંત્રણ સલાહ અનુસરો.",
"prevention": "હવાનો પ્રવાહ સુધારો, બિનજરૂરી પાન ભીનાશ ટાળો, નિયમિત તપાસ કરો અને સ્વસ્થ રોપણી સામગ્રી વાપરો."
}},

"Tomato_Leaf_Mold": {
"English": {
"disease": "Tomato Leaf Mold",
"crop": "Tomato",
"symptoms": "Yellow patches may appear on the upper leaf surface while olive-green to brown fungal growth can develop underneath.",
"cause": "Associated with the fungus Passalora fulva. High humidity and poor air circulation can favor the disease.",
"treatment": "Remove severely affected leaves where practical and improve ventilation. Reduce humidity and prolonged leaf wetness. Follow locally approved fungicide guidance if needed.",
"prevention": "Provide good spacing and airflow, avoid excessive humidity and remove infected plant debris."
},
"Hindi": {
"disease": "टमाटर का पत्ती मोल्ड रोग",
"crop": "टमाटर",
"symptoms": "पत्तियों की ऊपरी सतह पर पीले धब्बे और नीचे की ओर जैतूनी-हरे से भूरे रंग की फफूंद जैसी वृद्धि दिखाई दे सकती है।",
"cause": "यह Passalora fulva नामक फफूंद से जुड़ा है। अधिक नमी और खराब हवा का प्रवाह रोग को बढ़ावा दे सकता है।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित पत्तियाँ हटाएँ और वेंटिलेशन सुधारें। नमी और लंबे समय तक पत्ती गीलापन कम करें। आवश्यकता पर स्थानीय रूप से स्वीकृत फफूंदनाशी सलाह लें।",
"prevention": "उचित दूरी और हवा का प्रवाह रखें, अत्यधिक नमी से बचें और संक्रमित पौध अवशेष हटाएँ।"
},
"Gujarati": {
"disease": "ટામેટાનો પાનનો મોલ્ડ રોગ",
"crop": "ટામેટા",
"symptoms": "પાનની ઉપરની સપાટી પર પીળા ધબ્બા અને નીચેની બાજુ ઓલિવ-લીલીથી ભૂરા રંગની ફૂગ જેવી વૃદ્ધિ દેખાઈ શકે છે.",
"cause": "આ રોગ Passalora fulva નામની ફૂગ સાથે જોડાયેલો છે. વધુ ભેજ અને હવાના નબળા પ્રવાહથી રોગ વધી શકે છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત પાન દૂર કરો અને હવાની અવરજવર સુધારો. ભેજ અને લાંબા સમય સુધી પાન ભીનાશ ઘટાડો. જરૂર પડે તો સ્થાનિક રીતે માન્ય ફૂગનાશક અંગેની સલાહ લો.",
"prevention": "યોગ્ય અંતર અને હવાનો પ્રવાહ રાખો, વધારે ભેજ ટાળો અને ચેપગ્રસ્ત છોડના અવશેષો દૂર કરો."
}},

"Tomato_Septoria_leaf_spot": {
"English": {
"disease": "Tomato Septoria Leaf Spot",
"crop": "Tomato",
"symptoms": "Small circular spots with dark margins may develop on leaves, sometimes with tiny dark centers.",
"cause": "Caused by the fungus Septoria lycopersici. Moisture and splashing water can help spread the disease.",
"treatment": "Remove affected leaves where practical and improve airflow. Avoid splashing water onto foliage and maintain sanitation. Follow locally approved disease-control advice if needed.",
"prevention": "Use clean planting material, remove infected debris, avoid overhead watering and maintain good spacing."
},
"Hindi": {
"disease": "टमाटर का सेप्टोरिया पत्ती धब्बा",
"crop": "टमाटर",
"symptoms": "पत्तियों पर गहरे किनारों वाले छोटे गोल धब्बे बन सकते हैं। कभी-कभी इनके बीच छोटे गहरे बिंदु दिखाई देते हैं।",
"cause": "यह Septoria lycopersici नामक फफूंद से होता है। नमी और पानी की छींटें रोग को फैलाने में मदद कर सकती हैं।",
"treatment": "जहाँ संभव हो, प्रभावित पत्तियाँ हटाएँ और हवा का प्रवाह सुधारें। पत्तियों पर पानी की छींटें पड़ने से बचाएँ और स्वच्छता रखें। आवश्यकता पर स्थानीय रोग नियंत्रण सलाह लें।",
"prevention": "साफ रोपण सामग्री लें, संक्रमित अवशेष हटाएँ, ऊपर से पानी देने से बचें और उचित दूरी रखें।"
},
"Gujarati": {
"disease": "ટામેટાનો સેપ્ટોરિયા પાન ધબ્બા રોગ",
"crop": "ટામેટા",
"symptoms": "પાન પર ઘેરા કિનારા ધરાવતા નાના ગોળ ડાઘ થઈ શકે છે. ક્યારેક વચ્ચે નાના ઘેરા બિંદુઓ દેખાય છે.",
"cause": "આ રોગ Septoria lycopersici નામની ફૂગથી થાય છે. ભેજ અને પાણીના છાંટાથી રોગ ફેલાઈ શકે છે.",
"treatment": "શક્ય હોય ત્યાં અસરગ્રસ્ત પાન દૂર કરો અને હવાનો પ્રવાહ સુધારો. પાન પર પાણીના છાંટા પડતા ટાળો અને સ્વચ્છતા રાખો. જરૂર પડે તો સ્થાનિક રોગ નિયંત્રણ સલાહ લો.",
"prevention": "સ્વચ્છ રોપણી સામગ્રી વાપરો, ચેપગ્રસ્ત અવશેષો દૂર કરો, ઉપરથી પાણી આપવાનું ટાળો અને યોગ્ય અંતર રાખો."
}},

"Tomato_Spider_mites_Two_spotted_spider_mite": {
"English": {
"disease": "Tomato Two-Spotted Spider Mites",
"crop": "Tomato",
"symptoms": "Fine pale speckling or yellowing may appear on leaves. Heavy infestations can cause leaf bronzing, drying and fine webbing.",
"cause": "Caused by tiny spider mites, which are favored by hot and dry conditions.",
"treatment": "Inspect the undersides of leaves and separate heavily affected plants where practical. Reduce plant stress and use an appropriate locally approved mite-control method if required.",
"prevention": "Monitor leaves regularly, maintain adequate plant moisture and avoid conditions that encourage mite outbreaks."
},
"Hindi": {
"disease": "टमाटर में दो-धब्बेदार मकड़ी के कण",
"crop": "टमाटर",
"symptoms": "पत्तियों पर बहुत छोटे हल्के बिंदु या पीलापन दिखाई दे सकता है। अधिक संक्रमण में पत्तियाँ कांस्य रंग की, सूखी हो सकती हैं और महीन जाला दिखाई दे सकता है।",
"cause": "यह बहुत छोटे spider mites के कारण होता है। गर्म और शुष्क परिस्थितियाँ इनके बढ़ने के लिए अनुकूल होती हैं।",
"treatment": "पत्तियों की निचली सतह की जाँच करें और जहाँ संभव हो बहुत प्रभावित पौधों को अलग रखें। पौधे पर तनाव कम करें और आवश्यकता पर स्थानीय रूप से स्वीकृत mite-control उपाय अपनाएँ।",
"prevention": "पत्तियों की नियमित निगरानी करें, पौधों में पर्याप्त नमी रखें और ऐसे वातावरण से बचें जो mite outbreak को बढ़ावा दे।"
},
"Gujarati": {
"disease": "ટામેટામાં બે-ધબ્બાવાળા સ્પાઇડર માઇટ્સ",
"crop": "ટામેટા",
"symptoms": "પાન પર નાના ફિક્કા બિંદુઓ અથવા પીળાશ દેખાઈ શકે છે. વધુ ઉપદ્રવમાં પાન કાંસ્ય રંગના અથવા સૂકા થઈ શકે છે અને ઝીણું જાળું દેખાઈ શકે છે.",
"cause": "આ રોગ ખૂબ નાના spider mites ના ઉપદ્રવથી થાય છે. ગરમ અને સૂકી પરિસ્થિતિ તેમના વિકાસ માટે અનુકૂળ છે.",
"treatment": "પાનની નીચેની બાજુ તપાસો અને શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત છોડને અલગ રાખો. છોડ પરનો તણાવ ઘટાડો અને જરૂર પડે તો સ્થાનિક રીતે માન્ય mite-control પદ્ધતિ અપનાવો.",
"prevention": "પાનની નિયમિત તપાસ કરો, છોડમાં પૂરતી ભેજ રાખો અને માઇટના ઉપદ્રવને અનુકૂળ પરિસ્થિતિ ટાળો."
}},

"Tomato__Target_Spot": {
"English": {
"disease": "Tomato Target Spot",
"crop": "Tomato",
"symptoms": "Brown circular or irregular lesions can develop on leaves, often showing concentric ring patterns.",
"cause": "Associated with the fungus Corynespora cassiicola. Warm, humid conditions can favor disease development.",
"treatment": "Remove badly affected leaves where practical, improve airflow and avoid prolonged leaf wetness. Follow locally approved fungicide recommendations if required.",
"prevention": "Maintain spacing, improve ventilation, remove infected debris and avoid excessive leaf wetness."
},
"Hindi": {
"disease": "टमाटर का टार्गेट स्पॉट",
"crop": "टमाटर",
"symptoms": "पत्तियों पर भूरे गोल या अनियमित धब्बे बन सकते हैं जिनमें अक्सर गोल-गोल छल्ले दिखाई देते हैं।",
"cause": "यह Corynespora cassiicola नामक फफूंद से जुड़ा है। गर्म और नम परिस्थितियाँ रोग को बढ़ावा दे सकती हैं।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित पत्तियाँ हटाएँ, हवा का प्रवाह सुधारें और पत्तियों को लंबे समय तक गीला न रहने दें। आवश्यकता पर स्थानीय रूप से स्वीकृत फफूंदनाशी सलाह लें।",
"prevention": "उचित दूरी रखें, वेंटिलेशन सुधारें, संक्रमित अवशेष हटाएँ और अत्यधिक पत्ती गीलापन से बचें।"
},
"Gujarati": {
"disease": "ટામેટાનો ટાર્ગેટ સ્પોટ",
"crop": "ટામેટા",
"symptoms": "પાન પર ભૂરા ગોળ અથવા અનિયમિત ડાઘ થઈ શકે છે જેમાં ઘણી વખત ગોળ વલયો દેખાય છે.",
"cause": "આ રોગ Corynespora cassiicola નામની ફૂગ સાથે જોડાયેલો છે. ગરમ અને ભેજવાળી પરિસ્થિતિ રોગને અનુકૂળ છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત પાન દૂર કરો, હવાનો પ્રવાહ સુધારો અને પાન લાંબા સમય સુધી ભીના ન રહેવા દો. જરૂર પડે તો સ્થાનિક રીતે માન્ય ફૂગનાશક અંગેની સલાહ લો.",
"prevention": "યોગ્ય અંતર રાખો, હવાની અવરજવર સુધારો, ચેપગ્રસ્ત અવશેષો દૂર કરો અને વધારે પાન ભીનાશ ટાળો."
}},

"Tomato__Tomato_YellowLeaf__Curl_Virus": {
"English": {
"disease": "Tomato Yellow Leaf Curl Virus",
"crop": "Tomato",
"symptoms": "Leaves may curl upward, become yellow and remain smaller. Plants may show reduced growth and poor fruit development.",
"cause": "Caused by Tomato yellow leaf curl virus and commonly spread by whiteflies.",
"treatment": "Remove severely affected plants where practical to reduce sources of infection. Manage whiteflies using locally approved integrated pest-management methods. There is no simple cure that reverses virus infection.",
"prevention": "Use healthy seedlings, monitor and manage whiteflies, remove infected plants and control weeds that may support pests."
},
"Hindi": {
"disease": "टमाटर का पीला पत्ती मरोड़ विषाणु रोग",
"crop": "टमाटर",
"symptoms": "पत्तियाँ ऊपर की ओर मुड़ सकती हैं, पीली और छोटी रह सकती हैं। पौधे की वृद्धि कम हो सकती है और फल विकास प्रभावित हो सकता है।",
"cause": "यह Tomato yellow leaf curl virus से होता है और सामान्यतः सफेद मक्खी द्वारा फैलता है।",
"treatment": "जहाँ संभव हो, बहुत प्रभावित पौधों को हटाएँ ताकि संक्रमण के स्रोत कम हों। सफेद मक्खी का नियंत्रण स्थानीय रूप से स्वीकृत एकीकृत कीट प्रबंधन तरीकों से करें। वायरस संक्रमण को सामान्य उपचार से वापस ठीक नहीं किया जा सकता।",
"prevention": "स्वस्थ पौध लें, सफेद मक्खी की निगरानी व नियंत्रण करें, संक्रमित पौधे हटाएँ और कीटों को सहारा देने वाली खरपतवार नियंत्रित करें।"
},
"Gujarati": {
"disease": "ટામેટાનો પીળો પાન વળાંક વાયરસ રોગ",
"crop": "ટામેટા",
"symptoms": "પાન ઉપરની તરફ વળી શકે છે, પીળા અને નાના રહી શકે છે. છોડની વૃદ્ધિ ઘટી શકે છે અને ફળનો વિકાસ અસરગ્રસ્ત થઈ શકે છે.",
"cause": "આ રોગ Tomato yellow leaf curl virus થી થાય છે અને સામાન્ય રીતે સફેદ માખી દ્વારા ફેલાય છે.",
"treatment": "શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત છોડ દૂર કરો જેથી ચેપના સ્ત્રોત ઘટે. સફેદ માખીનું નિયંત્રણ સ્થાનિક રીતે માન્ય સંકલિત જીવાત વ્યવસ્થાપન પદ્ધતિથી કરો. વાયરસના ચેપને સામાન્ય સારવારથી પાછો ઠીક કરી શકાતો નથી.",
"prevention": "સ્વસ્થ રોપા વાપરો, સફેદ માખીની નિયમિત તપાસ અને નિયંત્રણ કરો, ચેપગ્રસ્ત છોડ દૂર કરો અને જીવાતોને આશ્રય આપતી નીંદણ નિયંત્રિત કરો."
}},

"Tomato__Tomato_mosaic_virus": {
"English": {
"disease": "Tomato Mosaic Virus",
"crop": "Tomato",
"symptoms": "Leaves may show mottled light and dark green patterns, distortion or reduced growth. Fruit development can also be affected.",
"cause": "Caused by Tomato mosaic virus. It can spread through infected plant material, contaminated hands and tools.",
"treatment": "There is no simple cure that removes the virus from an infected plant. Remove severely affected plants where practical and sanitize hands and tools.",
"prevention": "Use healthy planting material, clean tools, wash hands after handling plants and avoid using infected plant material for propagation."
},
"Hindi": {
"disease": "टमाटर का मोज़ेक विषाणु रोग",
"crop": "टमाटर",
"symptoms": "पत्तियों पर हल्के और गहरे हरे रंग का चितकबरा पैटर्न, विकृति या कम वृद्धि दिखाई दे सकती है। फल विकास भी प्रभावित हो सकता है।",
"cause": "यह Tomato mosaic virus से होता है और संक्रमित पौध सामग्री, दूषित हाथों और औजारों से फैल सकता है।",
"treatment": "संक्रमित पौधे से वायरस हटाने वाला कोई सरल उपचार नहीं है। जहाँ संभव हो, बहुत प्रभावित पौधे हटाएँ और हाथ तथा औजार साफ करें।",
"prevention": "स्वस्थ रोपण सामग्री लें, औजार साफ रखें, पौधों को छूने के बाद हाथ धोएँ और संक्रमित पौध सामग्री से नए पौधे न तैयार करें।"
},
"Gujarati": {
"disease": "ટામેટાનો મોઝેક વાયરસ રોગ",
"crop": "ટામેટા",
"symptoms": "પાન પર હળવા અને ઘેરા લીલા રંગનું ચિતરાયેલું પેટર્ન, વિકૃતિ અથવા ઓછી વૃદ્ધિ દેખાઈ શકે છે. ફળનો વિકાસ પણ અસરગ્રસ્ત થઈ શકે છે.",
"cause": "આ રોગ Tomato mosaic virus થી થાય છે અને ચેપગ્રસ્ત છોડની સામગ્રી, દૂષિત હાથ અને સાધનો દ્વારા ફેલાઈ શકે છે.",
"treatment": "ચેપગ્રસ્ત છોડમાંથી વાયરસ દૂર કરતું કોઈ સરળ ઉપચાર નથી. શક્ય હોય ત્યાં વધુ અસરગ્રસ્ત છોડ દૂર કરો અને હાથ તથા સાધનો સાફ રાખો.",
"prevention": "સ્વસ્થ રોપણી સામગ્રી વાપરો, સાધનો સાફ રાખો, છોડને સ્પર્શ કર્યા પછી હાથ ધોવો અને ચેપગ્રસ્ત છોડની સામગ્રીથી નવા છોડ તૈયાર ન કરો."
}},

"Tomato_healthy": {
"English": {
"disease": "Healthy Tomato",
"crop": "Tomato",
"symptoms": "No major disease pattern was detected in the uploaded image.",
"cause": "No clear disease pattern detected.",
"treatment": "No disease-specific treatment is indicated by this screening result. Continue normal crop care and monitoring.",
"prevention": "Maintain good spacing, balanced nutrition, appropriate watering and regular inspection."
},
"Hindi": {
"disease": "स्वस्थ टमाटर",
"crop": "टमाटर",
"symptoms": "अपलोड की गई तस्वीर में किसी प्रमुख रोग का पैटर्न नहीं मिला।",
"cause": "कोई स्पष्ट रोग पैटर्न नहीं मिला।",
"treatment": "इस जाँच के आधार पर किसी विशेष रोग उपचार की आवश्यकता नहीं बताई गई है। सामान्य देखभाल और निगरानी जारी रखें।",
"prevention": "उचित दूरी, संतुलित पोषण, सही सिंचाई और नियमित निरीक्षण रखें।"
},
"Gujarati": {
"disease": "સ્વસ્થ ટામેટા",
"crop": "ટામેટા",
"symptoms": "અપલોડ કરેલી તસવીરમાં કોઈ મુખ્ય રોગનું પેટર્ન મળ્યું નથી.",
"cause": "કોઈ સ્પષ્ટ રોગનું પેટર્ન મળ્યું નથી.",
"treatment": "આ તપાસના આધારે કોઈ ખાસ રોગની સારવાર સૂચવાતી નથી. સામાન્ય સંભાળ અને દેખરેખ ચાલુ રાખો.",
"prevention": "યોગ્ય અંતર, સંતુલિત પોષણ, યોગ્ય સિંચાઈ અને નિયમિત તપાસ રાખો."
}}

}

# ============================================================
# LANGUAGE SELECTION
# ============================================================

st.sidebar.title("🌱 AgriVision")

language = st.sidebar.selectbox(
    "Language / भाषा / ભાષા",
    ["English", "Hindi", "Gujarati"]
)

T = UI[language]

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="brand">
    <h1>🌱 AgriVision</h1>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<div class='brand'><p>{T['subtitle']}</p></div>",
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(T["upload"])

st.caption(T["upload_help"])

uploaded_file = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    analyze = st.button(
        T["analyze"],
        use_container_width=True,
        type="primary"
    )

    if analyze:

        with st.spinner(T["analyzing"]):

            # Preprocessing
            img = image.resize((224, 224))
            arr = np.array(img).astype("float32")
            arr = np.expand_dims(arr, axis=0)

            # MobileNetV2 preprocessing
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

            # Prediction
            predictions = model.predict(arr, verbose=0)[0]

            top_indices = np.argsort(predictions)[::-1][:3]

            top_idx = top_indices[0]
            second_idx = top_indices[1]

            predicted_class = class_names[top_idx]

            score = float(predictions[top_idx])
            second_score = float(predictions[second_idx])
            margin = score - second_score

        # ====================================================
        # RESULT
        # ====================================================

        info = DISEASE_INFO.get(predicted_class, {}).get(language)

        st.markdown(f"## {T['result']}")

        if info:

            st.markdown(
                f"""
                <div class="result-box">
                    <h2>{info['disease']}</h2>
                    <p><b>{T['crop']}:</b> {info['crop']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # MODEL RELIABILITY
            # ------------------------------------------------

            if score >= 0.80 and margin >= 0.30:
                reliability = T["higher"]
            elif score >= 0.55 and margin >= 0.15:
                reliability = T["moderate"]
            else:
                reliability = T["verify"]

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="score">{reliability}</div>
                    <p>{T['score']}: <b>{score*100:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # SYMPTOMS
            # ------------------------------------------------

            with st.expander(T["symptoms"], expanded=True):
                st.write(info["symptoms"])

            # ------------------------------------------------
            # CAUSE
            # ------------------------------------------------

            with st.expander(T["causes"], expanded=True):
                st.write(info["cause"])

            # ------------------------------------------------
            # TREATMENT
            # ------------------------------------------------

            with st.expander(T["treatment"], expanded=True):
                st.write(info["treatment"])

            # ------------------------------------------------
            # PREVENTION
            # ------------------------------------------------

            with st.expander(T["prevention"], expanded=True):
                st.write(info["prevention"])

            # ------------------------------------------------
            # NEXT STEP
            # ------------------------------------------------

            st.info(
                f"**{T['next_step']}:** "
                + (
                    info["treatment"]
                    if predicted_class not in [
                        "Pepper__bell___healthy",
                        "Potato___healthy",
                        "Tomato_healthy"
                    ]
                    else T["healthy_note"]
                )
            )

        else:
            st.warning(T["not_available"])

        # ====================================================
        # ALTERNATIVE PREDICTIONS
        # ====================================================

        with st.expander(T["alternatives"]):

            for rank, idx in enumerate(top_indices, start=1):

                cls = class_names[idx]
                pct = float(predictions[idx]) * 100

                alt_info = DISEASE_INFO.get(cls, {}).get(language)

                if alt_info:
                    display_name = alt_info["disease"]
                else:
                    display_name = cls

                st.write(
                    f"**{rank}. {display_name} — {pct:.2f}%**"
                )

        # ====================================================
        # TECHNICAL LABEL
        # ====================================================

        with st.expander(T["technical"]):
            st.code(predicted_class)

        # ====================================================
        # SAFETY
        # ====================================================

        st.warning(
            f"{T['warning']}\n\n"
            f"{T['safety']}\n\n"
            f"{T['wrong_warning']}"
        )

# ============================================================
# ABOUT
# ============================================================

st.divider()

with st.expander(f"ℹ️ {T['about']}"):

    st.write(T["about_text"])

    st.write(f"**{T['supported']}**")
    st.write(f"**{T['model']}**")
    st.write(f"**{T['classes']}**")

    st.caption(
        "AgriVision • AI for Smarter Crop Health"
    )
