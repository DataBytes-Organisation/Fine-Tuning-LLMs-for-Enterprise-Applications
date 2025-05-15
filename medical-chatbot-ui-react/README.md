# DataBytes Medical Chatbot UI

## About the Project

This project is a user interface for the DataBytes Personalised Medical chatbot. The chatbot provides tailored medical advice and support, designed to assist users with care and precision. It features a clean and intuitive design, ensuring a user-friendly experience while maintaining security and confidentiality.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

# Firebase Hosting
This project uses Google Firebase Hosting. 

https://medical-chatbot-ui.web.app/

To deploy use:
```
npm run build
firebase deploy --only hosting
```