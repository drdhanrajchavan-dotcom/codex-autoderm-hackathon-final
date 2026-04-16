import type { GetServerSideProps } from "next";

export const getServerSideProps: GetServerSideProps = async () => ({
  redirect: {
    destination: "/doctor",
    permanent: false,
  },
});

export default function HomePage() {
  return null;
}
