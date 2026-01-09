'use server';

/**
 * @fileOverview Summarizes key findings from an X-ray image analysis for dentists.
 *
 * - summarizeXrayFindings - A function that summarizes X-ray findings.
 * - SummarizeXrayFindingsInput - The input type for the summarizeXrayFindings function.
 * - SummarizeXrayFindingsOutput - The return type for the summarizeXrayFindings function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SummarizeXrayFindingsInputSchema = z.object({
  xrayDataUri: z
    .string()
    .describe(
      "An X-ray image as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  analysisResults: z
    .string()
    .describe('The detailed analysis results from the X-ray image processing.'),
});
export type SummarizeXrayFindingsInput = z.infer<
  typeof SummarizeXrayFindingsInputSchema
>;

const SummarizeXrayFindingsOutputSchema = z.object({
  summary: z.string().describe('A concise summary of the key findings.'),
});
export type SummarizeXrayFindingsOutput = z.infer<
  typeof SummarizeXrayFindingsOutputSchema
>;

export async function summarizeXrayFindings(
  input: SummarizeXrayFindingsInput
): Promise<SummarizeXrayFindingsOutput> {
  return summarizeXrayFindingsFlow(input);
}

const summarizeXrayFindingsPrompt = ai.definePrompt({
  name: 'summarizeXrayFindingsPrompt',
  input: {schema: SummarizeXrayFindingsInputSchema},
  output: {schema: SummarizeXrayFindingsOutputSchema},
  prompt: `You are an expert dental assistant. Your task is to summarize the key findings from an X-ray analysis for a dentist. Be concise and focus on the most important issues.

X-ray Image: {{media url=xrayDataUri}}
Analysis Results: {{{analysisResults}}}`,
});

const summarizeXrayFindingsFlow = ai.defineFlow(
  {
    name: 'summarizeXrayFindingsFlow',
    inputSchema: SummarizeXrayFindingsInputSchema,
    outputSchema: SummarizeXrayFindingsOutputSchema,
  },
  async input => {
    const {output} = await summarizeXrayFindingsPrompt(input);
    return output!;
  }
);
