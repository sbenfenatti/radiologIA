# **App Name**: RadiologIA

## Core Features:

- Secure Login: Provide a secure authentication mechanism for dentists to access the application.
- Epidemiological Dashboard: Display summarized, anonymized statistical data from analyzed X-rays using charts and graphs, giving dentists insights into common issues. Implement user controls to let the user visualize different parameters in the data.
- Group X-ray Analysis (Triage): Enable dentists to upload and process multiple X-rays simultaneously. Results are presented in a sortable list/table with key findings, and a traffic-light scheme, for efficient screening and triage. Also include a button for a follow-up action on individual findings.
- Individual X-ray Analysis (Diagnostic Aid): Provide a workspace for detailed X-ray analysis. Features include AI overlays (segmentation/detection), zoom and pan, a history or versions list, and the AI chat interface.
- AI Chat Assistant: Integrate a chat interface powered by generative AI to provide assistance with X-ray interpretation. The AI can summarize findings, suggest possible diagnoses, and answer questions based on the image data and dental knowledge; the responses will incorporate relevant data when appropriate, making the chatbot act like a tool in the dentist's decision-making process.
- Python Backend Integration: Establish a connection to the Python backend for X-ray image processing and AI analysis. This connection uses a secure protocol with appropriate API endpoints.

## Style Guidelines:

- Primary color: Calming, trustworthy blue (#75A9FF) for a modern health-tech feel.
- Background color: Light blue (#D1E4FF), same hue as primary, for a clean backdrop.
- Accent color: Complementary purple (#A375FF), an analogous hue to the primary, for highlights and call-to-action buttons.
- Font: 'Inter', a grotesque sans-serif, will be used for both headlines and body text, since the app will not have large continuous blocks of text. Its objective, neutral styling contributes to the app's trustworthy, professional look.
- Employ clear, easily recognizable icons that represent dental concepts and AI functionalities.
- Maintain a clean and well-organized layout that helps facilitate navigation and reduces clutter. Utilize white space to effectively separate elements.
- Subtle animations and transitions will enhance user experience (e.g., loading animations during analysis, smooth transitions between views).