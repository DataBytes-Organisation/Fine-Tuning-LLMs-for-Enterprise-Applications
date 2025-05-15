export async function getChatResponse(userMessage: string): Promise<string> {
  // Mock response logic
  if (userMessage.toLowerCase().includes("hello")) {
    return "Hi there! How can I assist you today?";
  } else if (userMessage.toLowerCase().includes("help")) {
    return `
## How I Can Help You
Here are some things I can assist you with:
- **Answering questions** about our services.
- **Providing guidance** on how to use the app.
- **Offering support** for any issues you encounter.
This is a [link](https://github.com/remarkjs/react-markdown)

1.  List item one.

    List item one continued with a second paragraph followed by an
    Indented block.

        $ ls *.sh
        $ mv *.sh ~/tmp

    List item continued with a third paragraph.

2.  List item two continued with an open block.

    This paragraph is part of the preceding list item.

    1. This list is nested and does not require explicit item continuation.

       This paragraph is part of the preceding list item.

    2. List item b.

    This paragraph belongs to item two of the outer list.

Feel free to ask me anything!`;
  } else {
    return "I'm here to assist you with any questions you have.";
  }

  // Example for calling an external API (uncomment if needed):
  // const response = await fetch("https://api.example.com/chat", {
  //   method: "POST",
  //   headers: {
  //     "Content-Type": "application/json",
  //   },
  //   body: JSON.stringify({ message: userMessage }),
  // });
  // const data = await response.json();
  // return data.reply;
}