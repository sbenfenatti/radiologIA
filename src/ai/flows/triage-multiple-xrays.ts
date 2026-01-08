'use server';

/**
 * @fileOverview This file defines a Genkit flow for triaging multiple X-rays.
 *
 * The flow takes a list of X-ray data URIs as input, analyzes each X-ray for key findings,
 * and prioritizes them using a traffic-light scheme.
 *
 * @exports {triageMultipleXrays} - The main function to triage multiple X-rays.
 * @exports {TriageMultipleXraysInput} - The input type for the triageMultipleXrays function.
 * @exports {TriageMultipleXraysOutput} - The output type for the triageMultipleXrays function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const XrayInputSchema = z.object({
  xrayDataUri: z
    .string()
    .describe(
      'An X-ray image as a data URI that must include a MIME type and use Base64 encoding. Expected format: \'data:<mimetype>;base64,<encoded_data>\'.'
    ),
});

const TriageMultipleXraysInputSchema = z.object({
  xrays: z.array(XrayInputSchema).describe('An array of X-ray images to triage.'),
});
export type TriageMultipleXraysInput = z.infer<typeof TriageMultipleXraysInputSchema>;

const TriageResultSchema = z.object({
  keyFindings: z.string().describe('Key findings from the X-ray analysis.'),
  priority: z
    .enum(['high', 'medium', 'low'])
    .describe('Priority level based on the findings (high, medium, low).'),
});

const TriageMultipleXraysOutputSchema = z.array(TriageResultSchema);
export type TriageMultipleXraysOutput = z.infer<typeof TriageMultipleXraysOutputSchema>;

export async function triageMultipleXrays(input: TriageMultipleXraysInput): Promise<TriageMultipleXraysOutput> {
  return triageMultipleXraysFlow(input);
}

const triageXrayPrompt = ai.definePrompt({
  name: 'triageXrayPrompt',
  input: {schema: XrayInputSchema},
  output: {schema: TriageResultSchema},
  prompt: `You are an expert dental radiologist.

  Analyze the provided X-ray image and identify any key findings.
  Based on these findings, determine the priority level (high, medium, or low) for follow-up.

  Respond with key findings and assign a priority based on urgency:
  - High: Immediate attention required (e.g., significant infection, fracture).
  - Medium: Further evaluation needed (e.g., possible caries, early periodontal disease).
  - Low: Routine monitoring (e.g., minor anomalies, normal findings).

X-ray Image: {{media url=xrayDataUri}}

  Format your answer as a valid JSON object.
  `,
});

const triageMultipleXraysFlow = ai.defineFlow(
  {
    name: 'triageMultipleXraysFlow',
    inputSchema: TriageMultipleXraysInputSchema,
    outputSchema: TriageMultipleXraysOutputSchema,
  },
  async input => {
    const triageResults: TriageResultSchema[] = [];

    for (const xray of input.xrays) {
      const {output} = await triageXrayPrompt(xray);
      triageResults.push(output!);
    }

    return triageResults;
  }
);
