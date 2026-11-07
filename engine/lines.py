"""Test lines for say.py / speak.py / check_stream.py: (tag, speaker, description, text)."""
LINES = [  # our own test lines (en / hi / Hinglish / mixed)
    ("en", "Ira", "professional, Indian English accent, steady pace", "Your loan application has been approved. The amount will reach your account by tomorrow."),
    ("hi", "Ira", "happy, Hindi accent, steady pace", "नमस्ते! आपकी किस्त की तारीख अगले सोमवार है, कृपया समय पर भुगतान कर दीजिए।"),
    ("hinglish", "Aisha", "calm, Hindi accent, steady pace", "Haan ji, aapka EMI ka payment mil gaya hai, ab koi tension nahi."),
    ("mixed", "Siya", "excited, Hindi accent, fast pace", "आज की meeting में सबको हमारा idea पसंद आया <laugh> अब coffee मेरी तरफ़ से!"),
]
HARD = [  # harder lines: length (crosses KV buckets), numbers/money/dates, fast code-switching, emotion tags, questions
    ("hard_long_hi", "Zoya", "warm, Hindi accent, storytelling, steady pace",
     "बहुत समय पहले, एक छोटे से गाँव में रामू नाम का एक किसान रहता था। हर सुबह वो सूरज निकलने से पहले उठता, अपने बैलों को चारा देता "
     "और खेत की तरफ़ चल पड़ता। एक दिन उसे खेत में एक पुराना संदूक मिला, जिसके ऊपर अजीब से निशान बने हुए थे। उसने सोचा, क्या इसे खोलूँ या गाँव के मुखिया को बता दूँ?"),
    ("hard_numbers", "Ira", "professional, Indian English accent, clear, steady pace",
     "Your EMI of rupees 12,450 for loan number 4471-0932 is due on 15th March 2026. A late fee of 2.5 percent, that is rupees 311, "
     "will apply after 3 days. Call 1800-209-4455 between 9 AM and 6:30 PM for help."),
    ("hard_switch", "Aisha", "friendly, Hindi accent, fast pace",
     "Arre yaar, kal ki presentation ka kya hua? Maine slides bhej di thi na, but manager ne bola ki revenue wala graph thoda confusing hai, "
     "toh honestly mujhe lagta hai we should redo it, warna Monday ko phir se wahi drama hoga."),
    ("hard_emotion", "Siya", "emotional, Hindi accent, slow pace",
     "मुझे विश्वास नहीं हो रहा... <sigh> पाँच साल बाद आज मैं अपने घर वापस आई हूँ। <laugh> माँ ने वही पुरानी चाय बनाई, "
     "और सच कहूँ तो, it felt like nothing had changed at all. क्या तुम्हें भी कभी ऐसा लगता है?"),
]
